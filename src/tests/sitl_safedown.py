#!/usr/bin/env python3
"""SafeDown against real ArduCopter: the app flies SITL, launched from the stand.

Each scenario boots ArduCopter SITL (sitl_launch.py) with the user's FC
parameters and the JSON physics (sitl_physics.py): mass, CG offset, wind and
gusts, the tilted 4-leg stand.  The real app (AcroCommander + Mission over
MAVLink, through the fault-injecting sitl_proxy.py) arms in ACRO, launches
WASP with GO, then SafeDown is triggered by the phone's button or the pilot's
switch, optionally after an upset or with a fault, and is watched to the
ground.  Metrics come from the physics' true state, not from the FC.

    python sitl_safedown.py              # every scenario, 3 at a time
    python sitl_safedown.py -k wind -j 2 # only names containing "wind"
    python sitl_safedown.py --list

Results: a table on stdout and sitl_safedown_results.json in --out.
"""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from concurrent.futures import as_completed
from dataclasses import dataclass, field, asdict
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ACRO, STABILIZE, ALT_HOLD, RTL, LAND = 1, 0, 2, 6, 9
# The FC with its GPS in the EKF and a compass for the heading.
GPS_PARAMS = dict(EK3_SRC1_POSXY=3, EK3_SRC1_VELXY=3, COMPASS_ENABLE=1)
HOME_RADIUS_M = 6.0
SWITCH = dict(down=1000, middle=1500, up=1900)


@dataclass
class Scenario:
    name: str
    trigger: str = "button"          # "button" (phone), "switch" (ch6 up, ALT_HOLD), "catch" (ch6 middle),
                                     # "fence": none, the drone flies out of the geofence
    arm_by_app: bool = False         # the phone's ARM (the arm switch's function over MAVLink)
    fence_m: float = None            # a geofence square this wide around the stand
    fence_circle_m: float = None     # radius of a 24-point circle, the phone's Make button
    rc_fail: int = 0                 # trigger "rcloss": SIM_RC_FAIL (1 no pulses, 2 receiver failsafe values)
    gps: bool = True                 # False: no GPS in the EKF, SafeDown lands where it is
    gps_loss: bool = False           # the GPS dies on the way home
    after_s: float = 1.5             # launch -> trigger
    mass_kg: float = None            # None: the physics default
    cg_offset_m: tuple = None        # None: the physics default
    wind_ned: tuple = (0.0, 0.0, 0.0)
    gust_sigma: float = 0.0
    wind_on_stand: bool = False      # False: the wind starts once it is off the stand
    stand_pitch_deg: float = 45.0
    stand_roll_deg: float = 0.0
    upset: tuple = None              # body torque (N m, 3) for upset_s at the trigger
    upset_s: float = 0.15
    pre_upset: tuple = None          # (torque N m, seconds before the trigger, duration s): a dive
    deny: tuple = ()                 # modes the proxy refuses: "ALT_HOLD", "LAND", "STABILIZE"
    freeze_at: str = None            # "level" / "return" / "descend": the Jetson hangs then
    weak_motor: tuple = None         # (motor 1..4, thrust scale)
    plan_b_after_s: float = None     # pilot: throttle mid, switch MIDDLE this long after the trigger
    steer: bool = False              # WASP steering at a test target before the trigger
    stick: str = None                # None: the trigger's own sticks. "before": throttle mid, then the
                                     # switch. "after": the switch first; throttle mid only once SafeDown
                                     # has the sticks (ALT_HOLD) or once the catch has given them back.
    stand_height_m: float = None     # stand top above the ground (physics default 0.4)
    pilot_mid: bool = False          # fence: once airborne, throttle to mid and the switch stays down
    flight_elev_deg: float = None    # fence: after liftoff, pitch to this elevation, then cross the fence
    stab_speed_mps: float = None     # catch: throttle mid once airborne, STAB once this fast
                                     # (after_s is then the longest wait for that speed)
    fence_wait_s: float = 15.0       # how long a fence trigger waits to leave the polygon
    watch_s: float = 120.0           # how long to follow the flight after the trigger
    takeover_after_s: float = None   # trigger "throttle_up": push pilot throttle to max at this time
    jetson_dead_after_s: float = None # trigger "jetson_dead": freeze proxy at this time (GCS failsafe)
    expect: str = None               # "home" (GPS) / "landed" (no GPS) by default; "pilot", "failed", "any"
    extra_params: dict = field(default_factory=dict)


def _wind(speed, deg):
    return (speed*math.cos(math.radians(deg)), speed*math.sin(math.radians(deg)), 0.0)


def scenarios():
    out = []
    add = out.append
    for trigger in ("button", "switch"):
        add(Scenario(f"nominal-{trigger}", trigger=trigger))
    for after in (0.6, 3.0, 5.0):
        add(Scenario(f"trigger-{after:g}s", after_s=after))
    for pitch in (15, 30):
        add(Scenario(f"stand-{pitch}deg", stand_pitch_deg=pitch))
    add(Scenario("stand-45deg-rolled-15", stand_roll_deg=15))
    # Near-vertical launches the user explicitly asked about: 80 deg (stand
    # tilted 10 deg off vertical) and 70 deg (20 deg off), with UI SAFEDOWN
    # (land-in-place) and under light-to-moderate wind.
    add(Scenario("stand-10deg-button-calm", stand_pitch_deg=10))
    add(Scenario("stand-20deg-button-calm", stand_pitch_deg=20))
    add(Scenario("stand-10deg-button-wind5", stand_pitch_deg=10,
                 wind_ned=_wind(5, 45), gust_sigma=1.2))
    add(Scenario("stand-20deg-button-wind7", stand_pitch_deg=20,
                 wind_ned=_wind(7, 90), gust_sigma=1.5))
    # UI SAFEDOWN with no GPS: the drone should still land peacefully right
    # under itself (the land_only path through DESCEND).
    add(Scenario("stand-10deg-button-nogps", stand_pitch_deg=10, gps=False,
                 expect="landed"))
    for mass in (1.3, 1.8):
        add(Scenario(f"mass-{mass:g}kg-switch", mass_kg=mass, trigger="switch"))
    for name, cg, stand in (("cg-fwd-2cm-stand30", (.02, 0, .01), 30), ("cg-aft-2cm", (-.02, 0, .01), 45),
                            ("cg-right-2.5cm", (0, .025, .01), 45)):
        add(Scenario(name, cg_offset_m=cg, stand_pitch_deg=stand))
    for speed, deg, gust in ((5, 0, 0), (5, 180, 0), (8, 90, 2.0), (10, 45, 3.0)):
        add(Scenario(f"wind-{speed}ms-{deg}deg" + (f"-gust{gust:g}" if gust else ""),
                     wind_ned=_wind(speed, deg), gust_sigma=gust))
    add(Scenario("wind-8ms-heavy-switch", wind_ned=_wind(8, 270), gust_sigma=2.0, mass_kg=1.5, trigger="switch"))
    # A hit or a prop strike at the trigger: about 1500 deg/s peaks.
    for name, torque, secs in (("upset-roll", (2.2, 0, 0), .15), ("upset-pitch-flip", (0, -2.2, 0), .2),
                               ("upset-tumble", (1.6, 1.6, .4), .2)):
        add(Scenario(name, upset=torque, upset_s=secs))
    add(Scenario("upset-roll-low", upset=(2.2, 0, 0), upset_s=.15, after_s=.7))
    add(Scenario("dive-steep", after_s=3.0, pre_upset=((0, -2.2, 0), 1.2, .2)))
    add(Scenario("dive-inverted-wind-switch", after_s=3.0, pre_upset=((0, -2.2, 0), 1.0, .3),
                 wind_ned=_wind(6, 120), gust_sigma=1.5, trigger="switch"))
    add(Scenario("extreme-tumble-ekf-breaks", after_s=3.0, pre_upset=((0, -6.0, 0), 1.2, .3), expect="any"))
    add(Scenario("steering-then-safedown", steer=True, after_s=2.5))
    add(Scenario("weak-motor-15pct", weak_motor=(1, .85)))
    add(Scenario("rtl-fast-10ms", extra_params=dict(RTL_SPEED=1000), trigger="switch"))
    # Faults: no way home, a dying GPS, refused modes, a hung Jetson, the pilot.
    add(Scenario("fault-rtl-refused-lands-here", deny=("RTL",), expect="landed"))
    add(Scenario("fault-gps-lost-on-the-way", gps_loss=True, expect="landed"))
    add(Scenario("fault-alt-hold-refused", deny=("ALT_HOLD",)))
    add(Scenario("fault-jetson-hangs-on-the-way", freeze_at="return"))
    add(Scenario("fault-jetson-hangs-while-levelling", freeze_at="level", expect="any"))
    add(Scenario("no-gps-lands-here", gps=False))
    add(Scenario("no-gps-switch-lands-here", gps=False, trigger="switch"))
    add(Scenario("plan-b-pilot-stabilize", plan_b_after_s=3.0, expect="caught"))
    # The pilot's STABILIZE: steadied for him, then his (throttle at mid).
    add(Scenario("arm-by-app-nominal", arm_by_app=True))
    add(Scenario("geofence-60m-straight", arm_by_app=True, fence_m=60, trigger="fence", after_s=.1))
    add(Scenario("geofence-100m-wind-5ms", arm_by_app=True, fence_m=100, trigger="fence", after_s=.1,
                 wind_ned=_wind(5, 0)))
    for fail, what in ((1, "no-pulses"), (2, "receiver-failsafe")):
        add(Scenario(f"radio-lost-{what}-rtl", trigger="rcloss", rc_fail=fail, after_s=2.0, arm_by_app=True,
                     extra_params=dict(FS_THR_ENABLE=1), expect="fc_home"))
        add(Scenario(f"radio-lost-{what}-now-land", trigger="rcloss", rc_fail=fail, after_s=2.0, arm_by_app=True,
                     expect="any"))
    add(Scenario("catch-nominal", trigger="catch", expect="caught"))
    add(Scenario("catch-low-0.7s", trigger="catch", after_s=.7, expect="caught"))
    add(Scenario("catch-tumble", trigger="catch", upset=(1.6, 1.6, .4), upset_s=.2, expect="caught"))
    add(Scenario("catch-dive", trigger="catch", after_s=3.0, pre_upset=((0, -2.2, 0), 1.2, .2), expect="caught"))
    add(Scenario("catch-heavy-wind", trigger="catch", mass_kg=1.5, wind_ned=_wind(8, 90), gust_sigma=2.0,
                 expect="caught"))
    # Pilot stick order. ALT_HOLD keeps the app's sticks for the whole return, so
    # the throttle stick is ignored either way. STABILIZE gives the sticks back
    # only once the drone is hovering: mid stick must already be there. The model
    # hovers at 12.5 % thrust, so these match mid-stick to that hover (the real
    # FC's learned MOT_THST_HOVER is 0.152; the pilot still holds the middle).
    # stand_pitch 20 is 70 degrees above the ground. The square is 200 m across,
    # so the nearest side is 100 m from the stand.
    # Launch elevation above the ground, 40 / 60 / 80 degrees. The stand is
    # nose-down by (90 - elevation): 50, 30 and 10. The shot stays in ACRO
    # with the throttle stick down; the pilot puts the stick at mid once it
    # is flying, and selects STAB only after it is fast. 27.8 m/s is 100 km/h.
    # Mid stick is the hover point (the model hovers at 12.5 % thrust).
    hover = dict(MOT_THST_HOVER=0.125)
    for elev, stand in ((40, 50), (60, 30), (80, 10)):
        add(Scenario(f"stab-hover-{elev}deg", trigger="catch", arm_by_app=True,
                     stand_pitch_deg=stand, stab_speed_mps=27.8, after_s=8,
                     expect="hover", watch_s=100, extra_params=dict(hover)))
    # 1.2 kg needs 16.7 % thrust. The FC still starts at 12.5 %, so mid stick
    # is light until the catch writes the thrust it measured.
    add(Scenario("stab-hover-heavy", trigger="catch", arm_by_app=True,
                 stand_pitch_deg=30, stab_speed_mps=27.8, after_s=8, mass_kg=1.2,
                 expect="hover", watch_s=140, extra_params=dict(hover)))
    add(Scenario("pilot-althold-stick-before", trigger="switch", stick="before"))
    add(Scenario("pilot-althold-stick-after", trigger="switch", stick="after"))
    add(Scenario("pilot-stab-stick-before", trigger="catch", stick="before", expect="caught",
                 extra_params=hover))
    add(Scenario("pilot-stab-stick-after", trigger="catch", stick="after", expect="measured",
                 extra_params=hover))
    add(Scenario("pilot-geofence-100m-elev70", arm_by_app=True, fence_m=200, trigger="fence",
                 after_s=.1, stand_pitch_deg=20, fence_wait_s=25, watch_s=220))
    # Geofence while WASP is still on the launch angle, at about 100 km/h.
    # Elevation 40 / 60 / 80 is stand pitch 50 / 30 / 10. The square is 40 m
    # across, so the side is 20 m from the stand: past the few metres it
    # takes to reach 27.8 m/s, and still pitched because ACRO is not levelling.
    # stab_speed_mps is the speed the breach has to be at; nobody selects STAB.
    for elev, stand in ((40, 50), (60, 30), (80, 10)):
        add(Scenario(f"fence-return-{elev}", arm_by_app=True, fence_m=40, trigger="fence",
                     after_s=.1, stand_pitch_deg=stand, stab_speed_mps=27.8,
                     fence_wait_s=40, watch_s=240))
    # Ten STAB catches and ten geofence returns. Each cell is one elevation
    # and one weight, and the shot has to be at 27.8 m/s (100 km/h). The
    # stand is nose-down by (90 - elevation). Mass None is the 0.9 kg model.
    for name, elev, stand, mass in (
            ("40-09", 40, 50, None), ("40-12", 40, 50, 1.2),
            ("50-10", 50, 40, 1.0), ("50-14", 50, 40, 1.4),
            ("60-09", 60, 30, None), ("60-11", 60, 30, 1.1),
            ("60-15", 60, 30, 1.5), ("70-12", 70, 20, 1.2),
            ("75-10", 75, 15, 1.0), ("80-13", 80, 10, 1.3)):
        add(Scenario(f"stab-matrix-{name}", trigger="catch", arm_by_app=True,
                     stand_pitch_deg=stand, stab_speed_mps=27.8, after_s=8,
                     mass_kg=mass, expect="hover", watch_s=140, extra_params=dict(hover)))
    for name, elev, stand, mass in (
            ("35-09", 35, 55, None), ("40-12", 40, 50, 1.2),
            ("45-10", 45, 45, 1.0), ("50-14", 50, 40, 1.4),
            ("55-11", 55, 35, 1.1), ("60-09", 60, 30, None),
            ("65-13", 65, 25, 1.3), ("70-15", 70, 20, 1.5),
            ("75-12", 75, 15, 1.2), ("80-10", 80, 10, 1.0)):
        add(Scenario(f"fence-matrix-{name}", arm_by_app=True, fence_m=40, trigger="fence",
                     after_s=.1, stand_pitch_deg=stand, stab_speed_mps=27.8, mass_kg=mass,
                     fence_wait_s=40, watch_s=260))
    # Already off the stand (launched steep). Then pitched to the angle in the
    # air, throttle left at mid, switch left down, and the 50 m circle is what
    # has to bring it home. The score is the recovery after the breach.
    for name, elev, mass, wind in (
            ("15", 15, None, None), ("30", 30, None, None), ("45", 45, None, None),
            ("60", 60, None, None), ("75", 75, None, None),
            ("30-m12", 30, 1.2, None), ("30-wind5", 30, None, _wind(5, 90))):
        extra = {} if wind is None else dict(wind_ned=wind, gust_sigma=1.5)
        add(Scenario(f"circle50-air-{name}", arm_by_app=True, fence_circle_m=50, trigger="fence",
                     after_s=.1, stand_pitch_deg=15, flight_elev_deg=elev, stab_speed_mps=15,
                     mass_kg=mass, pilot_mid=True, fence_wait_s=45, watch_s=260, **extra))
    # Controls for the field plan: the same 30 degree shot with nobody on the
    # throttle, and 35 / 45 degrees with the pilot at mid.
    add(Scenario("circle50-nopilot-30-09", arm_by_app=True, fence_circle_m=50, trigger="fence",
                 after_s=.1, stand_pitch_deg=60, stab_speed_mps=20, fence_wait_s=40, watch_s=260))
    for elev in (35, 45):
        add(Scenario(f"circle50-mid-{elev}-09", arm_by_app=True, fence_circle_m=50, trigger="fence",
                     after_s=.1, stand_pitch_deg=90-elev, stab_speed_mps=20, pilot_mid=True,
                     fence_wait_s=40, watch_s=260))
    # --- pilot-driven 40-scenario sweep --------------------------------
    # 10 per case, covering: throttle-up takeover, geofence breach,
    # Jetson death mid-air (FC GCS failsafe), and UI SAFEDOWN button.
    # Launch angles 80 deg (stand=10) and 70 deg (stand=20) are the core
    # geometry; each case also varies wind, mass, and trigger timing.
    for sc in _case_a_throttle_takeover(): add(sc)
    for sc in _case_b_geofence():          add(sc)
    for sc in _case_c_jetson_dead():       add(sc)
    for sc in _case_d_safedown_button():   add(sc)

    for sc in out:
        sc.expect = sc.expect or ("home" if sc.gps else "landed")
    return out


def _case_a_throttle_takeover():
    """Case A: pilot pushes throttle stick past PILOT_THR_TAKEOVER_FRAC mid-
    flight.  Expected: WASP releases every channel, FC stays in ACRO, drone
    flies the pilot's sticks.  No SafeDown runs."""
    out = []
    add = out.append
    add(Scenario("A1-throttle-takeover-vertical-calm",
                 trigger="throttle_up", stand_pitch_deg=0, takeover_after_s=3.0))
    add(Scenario("A2-throttle-takeover-80deg-calm",
                 trigger="throttle_up", stand_pitch_deg=10, takeover_after_s=3.0))
    add(Scenario("A3-throttle-takeover-70deg-calm",
                 trigger="throttle_up", stand_pitch_deg=20, takeover_after_s=3.0))
    add(Scenario("A4-throttle-takeover-80deg-wind5",
                 trigger="throttle_up", stand_pitch_deg=10, takeover_after_s=3.0,
                 wind_ned=_wind(5, 45), gust_sigma=1.0))
    add(Scenario("A5-throttle-takeover-70deg-wind10",
                 trigger="throttle_up", stand_pitch_deg=20, takeover_after_s=3.0,
                 wind_ned=_wind(10, 90), gust_sigma=1.5))
    add(Scenario("A6-throttle-takeover-80deg-wind15-gusty",
                 trigger="throttle_up", stand_pitch_deg=10, takeover_after_s=3.0,
                 wind_ned=_wind(15, 180), gust_sigma=3.0))
    add(Scenario("A7-throttle-takeover-late-8s",
                 trigger="throttle_up", stand_pitch_deg=10, takeover_after_s=8.0))
    add(Scenario("A8-throttle-takeover-early-0.5s",
                 trigger="throttle_up", stand_pitch_deg=10, takeover_after_s=0.5))
    add(Scenario("A9-throttle-takeover-heavy-mass",
                 trigger="throttle_up", stand_pitch_deg=10, takeover_after_s=3.0,
                 mass_kg=1.8))
    add(Scenario("A10-throttle-takeover-light-mass-upset",
                 trigger="throttle_up", stand_pitch_deg=10, takeover_after_s=3.0,
                 mass_kg=1.2, upset=(1.2, 0.8, 0.3), upset_s=0.2))
    for sc in out:
        sc.expect = "pilot"
    return out


def _case_b_geofence():
    """Case B: drone crosses a geofence in flight.  Expected: SafeDown
    stabilises, then RTL to the launch point (GPS + home required)."""
    out = []
    add = out.append
    add(Scenario("B1-fence-50m-calm-80deg",
                 arm_by_app=True, fence_m=50, trigger="fence", after_s=.1,
                 stand_pitch_deg=10, flight_elev_deg=30, stab_speed_mps=15,
                 fence_wait_s=25, watch_s=200))
    add(Scenario("B2-fence-100m-wind5-80deg",
                 arm_by_app=True, fence_m=100, trigger="fence", after_s=.1,
                 stand_pitch_deg=10, flight_elev_deg=30, stab_speed_mps=18,
                 wind_ned=_wind(5, 45), gust_sigma=1.0,
                 fence_wait_s=30, watch_s=220))
    add(Scenario("B3-fence-100m-wind10-70deg",
                 arm_by_app=True, fence_m=100, trigger="fence", after_s=.1,
                 stand_pitch_deg=20, flight_elev_deg=30, stab_speed_mps=18,
                 wind_ned=_wind(10, 90), gust_sigma=1.5,
                 fence_wait_s=30, watch_s=220))
    add(Scenario("B4-fence-100m-wind15-gusty-70deg",
                 arm_by_app=True, fence_m=100, trigger="fence", after_s=.1,
                 stand_pitch_deg=20, flight_elev_deg=35, stab_speed_mps=20,
                 wind_ned=_wind(15, 180), gust_sigma=3.0,
                 fence_wait_s=35, watch_s=240))
    add(Scenario("B5-fence-200m-fast-80deg",
                 arm_by_app=True, fence_m=200, trigger="fence", after_s=.1,
                 stand_pitch_deg=10, flight_elev_deg=45, stab_speed_mps=28,
                 fence_wait_s=35, watch_s=240))
    add(Scenario("B6-fence-circle100-wind10-80deg",
                 arm_by_app=True, fence_circle_m=100, trigger="fence", after_s=.1,
                 stand_pitch_deg=10, flight_elev_deg=30, stab_speed_mps=20,
                 wind_ned=_wind(10, 0), gust_sigma=1.2,
                 fence_wait_s=35, watch_s=240))
    add(Scenario("B7-fence-50m-near-vertical-wind7",
                 arm_by_app=True, fence_m=50, trigger="fence", after_s=.1,
                 stand_pitch_deg=5, flight_elev_deg=20, stab_speed_mps=12,
                 wind_ned=_wind(7, 90), gust_sigma=1.5,
                 fence_wait_s=25, watch_s=200))
    add(Scenario("B8-fence-200m-heavy-mass-wind10",
                 arm_by_app=True, fence_m=200, trigger="fence", after_s=.1,
                 stand_pitch_deg=20, flight_elev_deg=40, stab_speed_mps=25,
                 wind_ned=_wind(10, 45), gust_sigma=2.0, mass_kg=1.8,
                 fence_wait_s=35, watch_s=240))
    add(Scenario("B9-fence-100m-pilot-mid-80deg",
                 arm_by_app=True, fence_m=100, trigger="fence", after_s=.1,
                 stand_pitch_deg=10, flight_elev_deg=30, stab_speed_mps=18,
                 pilot_mid=True, fence_wait_s=30, watch_s=220))
    add(Scenario("B10-fence-100m-high-speed-35mps",
                 arm_by_app=True, fence_m=150, trigger="fence", after_s=.1,
                 stand_pitch_deg=10, flight_elev_deg=50, stab_speed_mps=35,
                 fence_wait_s=30, watch_s=240))
    for sc in out:
        sc.expect = "home"
    return out


def _case_c_jetson_dead():
    """Case C: the Jetson stops sending heartbeats mid-flight (cable pull,
    SoC crash, battery disconnect).  The FC's FS_GCS_ENABLE=1 must RTL on
    its own.  The scenario freezes the proxy so no Jetson packets reach the
    FC; the FC's GCS failsafe trips after FS_GCS_TIMEOUT seconds."""
    out = []
    add = out.append
    add(Scenario("C1-jetson-dead-calm-80deg",
                 trigger="jetson_dead", stand_pitch_deg=10, jetson_dead_after_s=2.0,
                 watch_s=90))
    add(Scenario("C2-jetson-dead-calm-70deg",
                 trigger="jetson_dead", stand_pitch_deg=20, jetson_dead_after_s=2.0,
                 watch_s=90))
    add(Scenario("C3-jetson-dead-wind5-80deg",
                 trigger="jetson_dead", stand_pitch_deg=10, jetson_dead_after_s=3.0,
                 wind_ned=_wind(5, 45), gust_sigma=1.0, watch_s=100))
    add(Scenario("C4-jetson-dead-wind10-70deg",
                 trigger="jetson_dead", stand_pitch_deg=20, jetson_dead_after_s=3.0,
                 wind_ned=_wind(10, 90), gust_sigma=1.5, watch_s=120))
    add(Scenario("C5-jetson-dead-wind15-gusty",
                 trigger="jetson_dead", stand_pitch_deg=10, jetson_dead_after_s=4.0,
                 wind_ned=_wind(15, 180), gust_sigma=3.0, watch_s=140))
    add(Scenario("C6-jetson-dead-late-8s",
                 trigger="jetson_dead", stand_pitch_deg=10, jetson_dead_after_s=8.0,
                 watch_s=140))
    add(Scenario("C7-jetson-dead-heavy-mass",
                 trigger="jetson_dead", stand_pitch_deg=20, jetson_dead_after_s=3.0,
                 mass_kg=1.8, watch_s=120))
    add(Scenario("C8-jetson-dead-upset",
                 trigger="jetson_dead", stand_pitch_deg=10, jetson_dead_after_s=2.5,
                 upset=(1.5, 1.0, 0.3), upset_s=0.2, watch_s=120))
    add(Scenario("C9-jetson-dead-no-compass",
                 trigger="jetson_dead", stand_pitch_deg=20, jetson_dead_after_s=3.0,
                 gps=False, expect="landed", watch_s=120))
    add(Scenario("C10-jetson-dead-at-speed",
                 trigger="jetson_dead", stand_pitch_deg=10, jetson_dead_after_s=5.0,
                 flight_elev_deg=45, stab_speed_mps=30, watch_s=140))
    for sc in out:
        sc.expect = sc.expect or "home"
    return out


def _case_d_safedown_button():
    """Case D: pilot taps SAFEDOWN in the app.  Expected: SafeDown runs its
    land_only path -- SWITCHING -> LEVEL -> STOP -> DESCEND (LAND).  No RTL,
    needs no GPS, no home.  Drone lands roughly under where it was."""
    out = []
    add = out.append
    add(Scenario("D1-safedown-calm-80deg",
                 trigger="button", stand_pitch_deg=10, after_s=2.0, watch_s=60))
    add(Scenario("D2-safedown-calm-70deg",
                 trigger="button", stand_pitch_deg=20, after_s=2.0, watch_s=60))
    add(Scenario("D3-safedown-wind5-80deg",
                 trigger="button", stand_pitch_deg=10, after_s=2.0,
                 wind_ned=_wind(5, 45), gust_sigma=1.0, watch_s=80))
    add(Scenario("D4-safedown-wind10-70deg",
                 trigger="button", stand_pitch_deg=20, after_s=2.5,
                 wind_ned=_wind(10, 90), gust_sigma=1.5, watch_s=100))
    add(Scenario("D5-safedown-wind15-gusty-80deg",
                 trigger="button", stand_pitch_deg=10, after_s=3.0,
                 wind_ned=_wind(15, 180), gust_sigma=3.0, watch_s=120))
    add(Scenario("D6-safedown-late-8s",
                 trigger="button", stand_pitch_deg=10, after_s=8.0, watch_s=120))
    add(Scenario("D7-safedown-heavy-mass",
                 trigger="button", stand_pitch_deg=20, after_s=2.5,
                 mass_kg=1.8, watch_s=100))
    add(Scenario("D8-safedown-upset-at-trigger",
                 trigger="button", stand_pitch_deg=10, after_s=2.0,
                 upset=(1.5, 1.0, 0.3), upset_s=0.2, watch_s=100))
    add(Scenario("D9-safedown-no-gps",
                 trigger="button", stand_pitch_deg=10, after_s=2.0,
                 gps=False, expect="landed", watch_s=80))
    add(Scenario("D10-safedown-at-speed-35mps",
                 trigger="button", stand_pitch_deg=10, after_s=4.0,
                 flight_elev_deg=50, stab_speed_mps=35, watch_s=120))
    for sc in out:
        sc.expect = sc.expect or "landed"  # land_only: lands where it is
    return out


# -- one run --------------------------------------------------------------------

class Run:
    def __init__(self, sc, instance, out_dir):
        self.sc, self.instance, self.out_dir = sc, instance, Path(out_dir)
        self.events = []
        self.samples = []

    def note(self, text):
        self.events.append((round(time.perf_counter()-self.t0, 2), text))

    def wait(self, predicate, timeout, what):
        end = time.perf_counter()+timeout
        while time.perf_counter() < end:
            if predicate():
                return True
            self.sample()
            time.sleep(.02)
        raise TimeoutError(what)

    def sample(self):
        st = self.sitl.physics.state()
        m = self.m
        b = self.c.backend
        att, hgt, acc = b.wasp_attitude(), b.height(max_age=.5), b.body_accel()
        self.samples.append(dict(
            st, physics_t=st["t"], t=time.perf_counter()-self.t0, phase=m.phase,
            stage=m.safedown.stage if m.safedown.active else "",
            fallback=m.safedown.fallback if m.safedown.active else None,
            mode=self.c.mode_name, armed=self.c.armed,
            fc_tilt=None if att is None else math.degrees(math.acos(max(-1, min(1, math.cos(att["roll"])*math.cos(att["pitch"]))))),
            fc_h=None if hgt is None else hgt[0], fc_climb=None if hgt is None else hgt[1],
            fc_accel_h=None if acc is None else math.hypot(acc[0], acc[1]),
            fc_pos_ok=b.position_ok()))
        return st

    def run(self):
        from sitl_launch import Sitl
        from sitl_physics import PhysicsConfig
        from sitl_proxy import MavProxy
        sc = self.sc
        opt = {k: v for k, v in (("mass_kg", sc.mass_kg), ("cg_offset_m", sc.cg_offset_m),
                                          ("stand_height_m", sc.stand_height_m)) if v is not None}
        if not sc.gps:
            os.environ["WASP_REQUIRE_GPS"] = "0"  # read when fc_control is imported
        calm = not sc.wind_on_stand
        # With GPS the drone powers up flat (the EKF will not start using GPS
        # tilted on the stand) and is put on the stand once it has a fix.
        flat_first = sc.gps
        cfg = PhysicsConfig(instance=self.instance, stand_pitch_deg=0.0 if flat_first else sc.stand_pitch_deg,
                            stand_roll_deg=sc.stand_roll_deg,
                            wind_ned=(0.0, 0.0, 0.0) if calm else sc.wind_ned,
                            gust_sigma=0.0 if calm else sc.gust_sigma,
                            seed=sum(map(ord, sc.name)), **opt)
        if not sc.gps:
            os.environ["WASP_REQUIRE_GPS"] = "0"
        from fc_control import AcroCommander, InterceptLaw
        from mission import phone_command
        params = dict(GPS_PARAMS if sc.gps else {}, **sc.extra_params)
        self.sitl = Sitl(self.instance, cfg, extra_params=params or None)
        logs = tempfile.TemporaryDirectory(prefix="sitl-safedown-")
        try:
            self.sitl.start()
            self.sitl.wait_ready()
            self.proxy = MavProxy(int(self.sitl.mavlink_url.rsplit(":", 1)[1]))
            self.proxy.deny_modes = {dict(ALT_HOLD=ALT_HOLD, LAND=LAND, STABILIZE=STABILIZE, RTL=RTL)[d]
                                     for d in sc.deny}
            self.c = AcroCommander(conn=self.proxy.url, transport="mavlink",
                                   axes=("roll", "pitch", "yaw", "throttle"),
                                   law=InterceptLaw(cam_tilt_deg=30), send_hz=50, mission=True,
                                   verbose=False, steer_modes={"ACRO", "ALT_HOLD"}, log_dir=logs.name)
            with contextlib.redirect_stdout(io.StringIO()), patch("mission._read_json", return_value={}):
                self.c.connect()
            self.m = m = self.c.mission
            for attr in ("idle_path", "wasp_path", "auto_path", "levels_path", "flight_path", "fence_path"):
                setattr(m, attr, str(Path(logs.name)/f"{attr}.json"))
            m.log = lambda text: self.note(text)
            self.t0 = time.perf_counter()
            cmd = lambda k, v: phone_command(m, k, v, time.perf_counter())
            for k, v in (("flightmode", "wasp"), ("mount", "90"), ("waspp", "launch_s:0"),
                         ("waspp", "throttle:1"), ("waspp", "ramp:0.1"), ("waspp", "straight:0.5")):
                why = cmd(k, v)
                if why:
                    raise RuntimeError(f"{k}={v}: {why}")
            if m.confirm_mount(90):
                raise RuntimeError("camera mount")
            if flat_first:
                # The FC needs its GPS position before arming, so home is
                # the stand; the app refuses GO without both anyway.
                self.wait(lambda: self.c.backend.position_ok(), 150, "EKF GPS position")
                self.sitl.physics.event("stand_tilt", deg=sc.stand_pitch_deg, deg_s=15)
                self.wait(lambda: abs(self.sitl.physics.state()["tilt"] - math.hypot(
                    sc.stand_pitch_deg, sc.stand_roll_deg)) < 3, 20, "drone on the stand")
                time.sleep(2)
                if not self.c.backend.position_ok():
                    raise RuntimeError("lost the GPS position when put on the stand")
            if sc.fence_circle_m or sc.fence_m:
                lat, lon = self.c.backend.position()
                if sc.fence_circle_m:
                    r = sc.fence_circle_m
                    pts = []
                    for i in range(24):
                        a = i/24*2*math.pi
                        pts.append([lat + r*math.cos(a)/111320,
                                    lon + r*math.sin(a)/(111320*math.cos(math.radians(lat)))])
                else:
                    dlat, dlon = sc.fence_m/2/111320, sc.fence_m/2/(111320*math.cos(math.radians(lat)))
                    pts = [[lat-dlat, lon-dlon], [lat+dlat, lon-dlon],
                           [lat+dlat, lon+dlon], [lat-dlat, lon+dlon]]
                why = cmd("fence", "set:" + json.dumps(pts))
                if why:
                    raise RuntimeError(f"geofence: {why}")
            if sc.arm_by_app:
                self.wait(lambda: not self.c.mission_state()["wasp_arm_why"], 20,
                          f"ARM ready ({self.c.mission_state()['wasp_arm_why']})")
                if sc.fence_m or sc.fence_circle_m:
                    self.wait(lambda: m._fence_state[0] is True, 10,
                              f"inside the fence (state {m._fence_state}, position {self.c.backend.position()})")
                why = cmd("arm", "1")
                if why:
                    raise RuntimeError(f"ARM refused: {why}")
                self.note("ARM from the app")
            else:
                self.sitl.arm()
            self.wait(lambda: self.c.armed and self.c.mode_name == "ACRO" and m.phase == "WASP_READY",
                      20, f"WASP_READY (gate: {self.c._mission_why})")
            # ArduCopter holds the motors for 2 s after arming; GO comes later.
            armed = time.perf_counter()
            self.wait(lambda: time.perf_counter()-armed > 3, 4, "arming delay")
            if sc.steer:
                self._start_detections()
            if sc.weak_motor:
                self.sitl.physics.event("motor_scale", motor=sc.weak_motor[0], scale=sc.weak_motor[1])
            why = cmd("engage", "1")
            if why:
                raise RuntimeError(f"GO refused: {why}")
            self.wait(lambda: m.phase == "WASP_FLIGHT", 3, "launch")
            launched = time.perf_counter()
            if not sc.wind_on_stand and (any(sc.wind_ned) or sc.gust_sigma):
                with contextlib.suppress(TimeoutError):
                    self.wait(lambda: self.sample().get("left_stand"), 2, "liftoff")
                self.sitl.physics.event("wind", ned=sc.wind_ned)
                if sc.gust_sigma:
                    self.sitl.physics.event("gusts", sigma=sc.gust_sigma)
                self.note("WIND ON")
            if sc.pre_upset:
                torque, lead, duration = sc.pre_upset
                self.wait(lambda: time.perf_counter()-launched >= sc.after_s-lead, sc.after_s+1, "flight")
                self.sitl.physics.event("torque", nm=torque, duration=duration)
                self.note("PRE-UPSET")
            speed_catch = sc.trigger == "catch" and sc.stab_speed_mps
            if not speed_catch:
                self.wait(lambda: time.perf_counter()-launched >= sc.after_s, sc.after_s+1, "flight")
            st = self.sample()
            at_trigger = dict(st)
            self.lifted = any(x.get("left_stand") for x in self.samples)
            if sc.upset:
                self.sitl.physics.event("torque", nm=sc.upset, duration=sc.upset_s)
            if sc.trigger == "switch":
                if sc.stick == "before":
                    self.sitl.rc.set_stick("throttle", .5)
                    time.sleep(.3)
                    self.note("PILOT: throttle mid, then switch UP")
                elif sc.stick == "after":
                    self.note("PILOT: switch UP, throttle still down")
                self.sitl.rc.set(6, SWITCH["up"])
            elif sc.trigger == "catch" and speed_catch:
                # ACRO, throttle down, app ARM and GO already happened. As soon
                # as it is flying the pilot puts the throttle at mid. The app
                # keeps the launch. STAB comes only once it is fast.
                try:
                    self.wait(lambda: self.sample().get("left_stand"), 15, "airborne")
                except TimeoutError:
                    st = self.samples[-1] if self.samples else {}
                    raise TimeoutError(
                        "airborne"
                        f" h={st.get('height')} tilt={st.get('tilt')} left={st.get('left_stand')}"
                        f" crashed={st.get('crashed')} on_stand={st.get('on_stand')}"
                        f" thrust={st.get('thrust_frac')} vel={st.get('vel')}"
                        f" phase={self.m.phase} mode={self.c.mode_name}")
                self.sitl.rc.set_stick("throttle", .5)
                mid = self.sample()
                self.speed_at_mid = math.hypot(*mid["vel"])
                self.note(f"PILOT: throttle mid, still ACRO, {self.speed_at_mid:.1f} m/s")
                try:
                    self.wait(lambda: math.hypot(*self.sample()["vel"]) >= sc.stab_speed_mps,
                              sc.after_s, "target speed")
                    self.note("PILOT: STAB at the target speed")
                except TimeoutError:
                    seen = math.hypot(*self.sample()["vel"])
                    self.note(f"PILOT: STAB at {seen:.1f} m/s, below {sc.stab_speed_mps}")
                st = self.sample()
                at_trigger = dict(st)
                self.lifted = any(x.get("left_stand") for x in self.samples)
                self.sitl.rc.set(6, SWITCH["middle"])
            elif sc.trigger == "catch":
                if sc.stick == "after":
                    # Throttle stays at the bottom through the handover.
                    self.note("PILOT: switch MIDDLE, throttle still down")
                    self.sitl.rc.set(6, SWITCH["middle"])
                else:
                    # The pilot: throttle to mid, then the switch to the middle.
                    self.sitl.rc.set_stick("throttle", .5)
                    if sc.stick == "before":
                        time.sleep(.3)
                        self.note("PILOT: throttle mid, then switch MIDDLE")
                    self.sitl.rc.set(6, SWITCH["middle"])
            elif sc.trigger == "rcloss":
                # The transmitter's link dies mid-flight: the FC's own radio failsafe.
                self.sitl.mav.mav.param_set_send(1, 1, b"SIM_RC_FAIL", sc.rc_fail, 9)
                self.note(f"RC LOST (SIM_RC_FAIL {sc.rc_fail})")
                with contextlib.suppress(TimeoutError):
                    self.wait(lambda: self.c.mode_name in ("RTL", "LAND"), 8, "radio failsafe")
                self.note(f"FC mode after the radio loss: {self.c.mode_name}")
            elif sc.trigger == "fence":
                # Nobody moves the mode switch. A pilot who has put the throttle
                # about mid leaves it there; the app still owns the sticks.
                if sc.pilot_mid:
                    try:
                        self.wait(lambda: self.sample().get("left_stand"), 15, "airborne")
                    except TimeoutError:
                        st = self.samples[-1] if self.samples else {}
                        raise TimeoutError(
                            "airborne"
                            f" h={st.get('height')} tilt={st.get('tilt')} left={st.get('left_stand')}"
                            f" crashed={st.get('crashed')} phase={self.m.phase} mode={self.c.mode_name}")
                    self.sitl.rc.set_stick("throttle", .5)
                    self.note("PILOT: throttle about mid, switch stays DOWN")
                if sc.flight_elev_deg is not None:
                    self._pitch_to_elevation(sc.flight_elev_deg)
                # Crossing the fence starts SafeDown. No further stick.
                try:
                    self.wait(lambda: m.safedown.active, sc.fence_wait_s, "SafeDown at the geofence")
                except TimeoutError:
                    st = self.samples[-1] if self.samples else {}
                    raise TimeoutError(
                        "SafeDown at the geofence"
                        f" h={st.get('height')} tilt={st.get('tilt')} left={st.get('left_stand')}"
                        f" crashed={st.get('crashed')} on_stand={st.get('on_stand')}"
                        f" phase={self.m.phase} mode={self.c.mode_name}"
                        f" fence={m._fence_state} pos={self.c.backend.position()}")
                if "GEOFENCE" not in m.safedown.reason:
                    raise RuntimeError(f"SafeDown, but not the fence: {m.safedown.reason}")
                at_trigger = dict(self.sample())
                self.lifted = any(x.get("left_stand") for x in self.samples)
                self.note(f"FENCE: {m.safedown.reason}")
            elif sc.trigger == "throttle_up":
                # Pilot pushes the throttle stick near max while mid-flight.
                # ArduPilot's RC_CHANNELS_RAW surfaces the raw stick even
                # under our override: fc_control latches _pilot_throttle_takeover
                # and the mission releases every channel, staying in ACRO.
                wait_s = sc.takeover_after_s if sc.takeover_after_s is not None else 3.0
                self.wait(lambda: self.sample().get("left_stand"), 10, "airborne")
                airborne_t = time.perf_counter()
                while time.perf_counter() - airborne_t < wait_s:
                    self.sample()
                    time.sleep(.05)
                self.sitl.rc.set_stick("throttle", 0.95)
                self.note(f"PILOT: throttle -> 0.95 at +{wait_s:g}s (takeover)")
                try:
                    self.wait(lambda: self.m.phase == "PILOT", 3.0, "WASP release")
                    self.note("WASP released: pilot in ACRO")
                except TimeoutError:
                    raise RuntimeError(
                        f"throttle-up did not release WASP (phase={self.m.phase})")
                at_trigger = dict(self.sample())
                self.lifted = any(x.get("left_stand") for x in self.samples)
            elif sc.trigger == "jetson_dead":
                # Simulated Jetson crash: freeze the proxy so no packets from
                # the Jetson reach the FC.  The FC must RTL on its own after
                # FS_GCS_TIMEOUT seconds (via FS_GCS_ENABLE=1 or 2).
                wait_s = sc.jetson_dead_after_s if sc.jetson_dead_after_s is not None else 3.0
                self.wait(lambda: self.sample().get("left_stand"), 10, "airborne")
                airborne_t = time.perf_counter()
                while time.perf_counter() - airborne_t < wait_s:
                    self.sample()
                    time.sleep(.05)
                self.proxy.frozen = True
                self.note(f"JETSON DEAD at +{wait_s:g}s: proxy frozen; FC on its own")
                try:
                    self.wait(lambda: self.c.mode_name in ("RTL", "LAND", "SMART_RTL"),
                              15, "FC GCS failsafe")
                    self.note(f"FC failsafe mode: {self.c.mode_name}")
                except TimeoutError:
                    raise RuntimeError(
                        "FC did not RTL after Jetson died: check FS_GCS_ENABLE")
                at_trigger = dict(self.sample())
                self.lifted = any(x.get("left_stand") for x in self.samples)
            else:
                if cmd("safedown", "1"):
                    raise RuntimeError("SAFEDOWN refused")
            self.trigger_t = time.perf_counter()
            self._watch(at_trigger)
        finally:
            self._stop_detections()
            for closer in (getattr(self, "c", None), getattr(self, "proxy", None)):
                with contextlib.suppress(Exception):
                    closer and closer.close()
            with contextlib.suppress(Exception):
                self.sitl.stop()
            logs.cleanup()
        return self.metrics(at_trigger)

    def _pitch_to_elevation(self, elev_deg):
        """Nose to `elev_deg` above the horizon while still inside the fence.

        ACRO is holding zero rate, so a short body torque changes the angle and
        the rate loop then holds the new one. The launch itself stays steep.
        """
        target = 90.0 - elev_deg
        deadline = time.perf_counter() + 6.0
        while time.perf_counter() < deadline:
            st = self.sample()
            rng = math.hypot(st["pos"][0], st["pos"][1])
            if st.get("crashed") or rng > 35 or st.get("height", 99) < 2:
                break
            err = target - st["tilt"]
            if abs(err) < 6 and abs(st["rates"][1]) < 80:
                break
            # A fifth-of-a-second shove, the same size as the dive tests. The
            # rate loop then stops the rotation and holds the new angle.
            ny = -2.2 if err > 0 else 2.2
            self.sitl.physics.event("torque", nm=(0.0, ny, 0.0), duration=0.18)
            time.sleep(0.28)
        st = self.sample()
        self.note(f"PILOT ANGLE: tilt {st['tilt']:.0f} deg, elevation {90-st['tilt']:.0f} deg, "
                  f"{math.hypot(*st['vel']):.0f} m/s, {math.hypot(st['pos'][0], st['pos'][1]):.0f} m out")

    def _watch(self, at_trigger):
        sc, m = self.sc, self.m
        frozen = plan_b = gps_dead = False
        returning = None
        stick_after = False
        release_t = None
        end = self.trigger_t + sc.watch_s
        while time.perf_counter() < end:
            st = self.sample()
            since = time.perf_counter() - self.trigger_t
            if m.safedown.stage == "RETURN" and returning is None:
                returning = time.perf_counter()
            if sc.gps_loss and not gps_dead and returning and time.perf_counter() - returning > 3:
                self.sitl.mav.mav.param_set_send(1, 1, b"SIM_GPS_DISABLE", 1, 9)
                gps_dead = True
                self.note("GPS DEAD")
            if sc.freeze_at and not frozen and (
                    (sc.freeze_at == "level" and m.safedown.stage in ("LEVEL", "STOP"))
                    or (sc.freeze_at == "return" and returning and time.perf_counter() - returning > 2)
                    or (sc.freeze_at == "descend" and m.safedown.stage == "DESCEND" and self.c.mode_name == "LAND")):
                self.proxy.frozen = frozen = True
                self.note(f"JETSON FROZEN in {m.safedown.stage}")
            if sc.plan_b_after_s is not None and not plan_b and since >= sc.plan_b_after_s:
                # The pilot's Plan B: throttle stick to mid first, then STABILIZE.
                self.sitl.rc.set_stick("throttle", .5)
                time.sleep(.2)
                self.sitl.rc.set(6, SWITCH["middle"])
                plan_b, self.plan_b_t = True, time.perf_counter()
                self.note("PILOT: throttle mid, switch MIDDLE")
            if plan_b and time.perf_counter() - self.plan_b_t > 8:
                break
            if sc.stick == "after" and sc.trigger == "switch" and not stick_after and since >= 0.4:
                self.sitl.rc.set_stick("throttle", .5)
                stick_after = True
                self.note("PILOT: throttle mid while SafeDown still holds the sticks")
            caught = any("CATCH done" in e for _t, e in self.events)
            if sc.stick == "after" and sc.trigger == "catch" and caught:
                # The app has given the sticks back and the throttle is still down.
                # Half a second later the pilot puts it at mid: too late for the handover.
                release_t = release_t or time.perf_counter()
                if not stick_after and time.perf_counter() - release_t >= 0.5:
                    self.sitl.rc.set_stick("throttle", .5)
                    stick_after = True
                    self.note("PILOT: throttle mid 0.5 s after the handover")
                if stick_after and time.perf_counter() - release_t > 6.5:
                    break
            if sc.expect in ("caught", "hover") and caught:
                self.caught_t = getattr(self, "caught_t", None) or time.perf_counter()
                if sc.expect == "hover" or sc.stick == "before":
                    # Exactly mid, the hover point. No extra correction.
                    self.sitl.rc.set_stick("throttle", .5)
                elif st.get("vel") is not None:
                    # The pilot flies it now: mid stick, plus what holds the height.
                    self._pilot_i = max(-.3, min(.3, getattr(self, "_pilot_i", 0.0) + .002*st["vel"][2]))
                    self.sitl.rc.set_stick("throttle", max(.2, min(.95, .5 + self._pilot_i + .06*st["vel"][2])))
                if time.perf_counter() - self.caught_t > (8 if sc.expect == "hover" else 6):
                    break
            if not self.c.armed and (st.get("on_ground") or st.get("crashed")):
                for _ in range(25):
                    self.sample()
                    time.sleep(.02)
                break
            if st.get("crashed"):
                break
            time.sleep(.02)

    # Fake detections for the "steering" scenario: a target off to one side.
    def _start_detections(self):
        import threading
        self._det_stop = threading.Event()

        def loop():
            while not self._det_stop.is_set():
                x, y = 120, -40
                cmd = self.c.compute(x, y, "LOCK", 640, 640, bbox=(0, 0, 24, 24),
                                     capture_t=time.perf_counter()-.03)
                self.c.update(cmd, (x, y), True, detected=True, conf=.9, bbox=(0, 0, 24, 24))
                time.sleep(.03)
        threading.Thread(target=loop, daemon=True).start()
        self.note("detections on")

    def _stop_detections(self):
        if getattr(self, "_det_stop", None):
            self._det_stop.set()

    def _caught_ok(self, s, crashed):
        # Handed over level and no longer shooting up or down, and still
        # flying under the pilot's mid stick a few seconds later.
        t = next((t for t, e in self.events if "CATCH done" in e), None)
        if t is None or crashed:
            return False
        at = next((x for x in s if x["t"] >= t), None)
        self.catch_report = None if at is None else dict(
            catch_s=round(t - (self.trigger_t - self.t0), 2), tilt=round(at["tilt"]), vz=round(-at["vel"][2], 1),
            vh=round(math.hypot(at["vel"][0], at["vel"][1]), 1), h=round(-at["pos"][2], 1))
        # The lean left at handover is the air brake's (at most 12 degrees plus
        # gusts): under the pilot's centred stick it is level a second later.
        later = next((x for x in s if x["t"] >= t + 1.0), None)
        return (at is not None and at["tilt"] < 20 and abs(at["vel"][2]) < 2.5
                and later is not None and later["tilt"] < 10)

    def _hover_ok(self, s, crashed):
        # Handed over stopped and level, and still up under the pilot's mid stick.
        t = next((t for t, e in self.events if "CATCH done" in e), None)
        if t is None or crashed or not self.lifted:
            return False
        at = next((x for x in s if x["t"] >= t), None)
        later = next((x for x in s if x["t"] >= t + 4.0), None)
        if at is None or later is None:
            self.catch_report = None
            return False
        end = next((x for x in s if x["t"] >= t + 7.5), None)
        if end is None:
            self.catch_report = None
            return False
        vh = math.hypot(at["vel"][0], at["vel"][1])
        later_vh = math.hypot(later["vel"][0], later["vel"][1])
        end_vh = math.hypot(end["vel"][0], end["vel"][1])
        h, later_h, end_h = -at["pos"][2], -later["pos"][2], -end["pos"][2]
        later_vz, end_vz = -later["vel"][2], -end["vel"][2]
        hold_drift = math.hypot(later["pos"][0]-at["pos"][0], later["pos"][1]-at["pos"][1])
        end_drift = math.hypot(end["pos"][0]-at["pos"][0], end["pos"][1]-at["pos"][1])
        self.catch_report = dict(
            catch_s=round(t - (self.trigger_t - self.t0), 2), tilt=round(at["tilt"]),
            vz=round(-at["vel"][2], 1), vh=round(vh, 1), h=round(h, 1),
            later_vh=round(later_vh, 1), later_vz=round(later_vz, 1),
            later_h=round(later_h, 1), drop_m=round(h - later_h, 1),
            hold_drift_m=round(hold_drift, 1),
            end_h=round(end_h, 1), end_vz=round(end_vz, 1), end_vh=round(end_vh, 1),
            end_drift_m=round(end_drift, 1), armed=bool(at.get("armed")),
            thrust=None if at.get("thrust_frac") is None else round(at["thrust_frac"], 4),
            later_thrust=None if later.get("thrust_frac") is None else round(later["thrust_frac"], 4),
            end_thrust=None if end.get("thrust_frac") is None else round(end["thrust_frac"], 4))
        # Mid stick has to hold the height both ways for the whole handover,
        # and the launch direction has to be gone. A slow sink that is still
        # inside the band at 4 s and running away at 8 s is not a hover.
        still = (at.get("armed") and not later.get("on_ground") and not later.get("crashed")
                 and not end.get("on_ground") and not end.get("crashed"))
        return (still and h > 5
                and vh <= 0.8 and abs(at["vel"][2]) <= 0.25 and at["tilt"] < 12
                and later_vh <= 0.6 and abs(later_vz) <= 0.20 and later["tilt"] < 12
                and abs(h - later_h) < 0.4 and hold_drift < 4.0
                and end_vh <= 0.6 and abs(end_vz) <= 0.25 and end["tilt"] < 12
                and abs(h - end_h) < 0.6 and end_drift < 5.0)

    def metrics(self, at_trigger):
        s = [x for x in self.samples if x["t"] >= self.trigger_t - self.t0]
        tilt = lambda x: x["tilt"]
        h = lambda x: -x["pos"][2]
        vh = lambda x: math.hypot(x["vel"][0], x["vel"][1])
        level_t = next((x["t"] for x in s if tilt(x) < 10 and x["t"] > s[0]["t"]+.05), None)
        descend = next((x for x in s if x["stage"] == "DESCEND"), None)
        touch = next((x for x in s if x.get("on_ground")), None)
        after_level = [x for x in s if level_t is not None and x["t"] >= level_t + 1.0 and not x.get("on_ground")]
        yaw_hold = None
        if descend:
            yaws = [x["euler"][2] for x in s if x["t"] >= descend["t"] and not x.get("on_ground")]
            if yaws:
                yaw_hold = max(abs((y - yaws[0] + 180) % 360 - 180) for y in yaws)
        events = [e for _t, e in self.events]
        handover_t = next((t for t, e in self.events if "CATCH done" in e), None)
        after_hand = [x for x in s if handover_t is not None and x["t"] >= handover_t]
        hand_h = None if not after_hand else h(after_hand[0])
        min_after = None if not after_hand else min(h(x) for x in after_hand)
        landed = any("SAFEDOWN complete: landed" in e for e in events)
        final = s[-1] if s else {}
        crashed = any(x.get("crashed") for x in s)
        tipped = bool(final) and final.get("on_ground") and tilt(final) > 45
        home_m = math.hypot(final["pos"][0], final["pos"][1]) if final else None
        returned = any(x["stage"] == "RETURN" for x in s)
        ok = {"landed": landed and not crashed and not tipped and self.lifted,
              "home": (landed and not crashed and not tipped and self.lifted and returned
                       and home_m is not None and home_m <= HOME_RADIUS_M),
              "pilot": not crashed and any("SAFEDOWN ended: pilot switched" in e for e in events),
              "failed": any("SAFEDOWN FAILED" in e for e in events),
              "caught": self._caught_ok(s, crashed),
              "hover": self._hover_ok(s, crashed),
              "measured": self.lifted and any("CATCH done" in e for e in events),
              "any": not crashed and not tipped,
              # The FC's own RTL (no SafeDown): back on the stand, disarmed, intact.
              "fc_home": (not crashed and not tipped and self.lifted and not self.c.armed
                          and bool(final.get("on_ground")) and home_m is not None and home_m <= HOME_RADIUS_M),
              }[self.sc.expect]
        if self.sc.gps_loss and not any(x.get("fc_pos_ok") is False for x in s):
            ok = False  # the GPS never went: the scenario tested nothing
        if not self.lifted:
            ok = None  # never left the stand: a launch finding, SafeDown never flew
        if self.sc.expect == "hover" and ok is None:
            ok = False
        tas = math.hypot(*at_trigger["vel"])
        if (self.sc.expect == "hover" and ok and self.sc.stab_speed_mps
                and tas < 0.95*self.sc.stab_speed_mps):
            ok = False  # never reached the speed this scenario exists to stop
        if self.sc.expect == "hover" and ok and not any("still ACRO" in e for e in events):
            ok = False  # STAB came before the pilot had put the throttle at mid, in ACRO
        # A fence return only counts if the breach was the fast, still-pitched
        # shot, and the way home did not tumble or hit hard.
        t_trig = self.trigger_t - self.t0
        breach = None
        if self.sc.trigger == "fence" and self.sc.stab_speed_mps:
            launch_tilt = (90 - self.sc.flight_elev_deg if self.sc.flight_elev_deg is not None
                           else self.sc.stand_pitch_deg)
            tilt_now = tilt(at_trigger)
            need = 0.95*self.sc.stab_speed_mps
            wild = max((tilt(x) for x in after_level), default=0)
            reasons = []
            if tas < need:
                reasons.append(f"speed {tas:.1f} m/s is under {need:.0f}")
            if abs(tilt_now - launch_tilt) > 25:
                reasons.append(f"tilt {tilt_now:.0f} deg has left the launch angle {launch_tilt:.0f}")
            if final and final.get("hard_landings"):
                reasons.append("hard landing")
            if wild > 60:
                reasons.append(f"tilt {wild:.0f} deg after it had levelled")
            if not self.lifted:
                reasons.append("never left the stand")
            if self.sc.fence_circle_m:
                # Steady means level and no longer climbing or diving. RTL may
                # still be fast horizontally: the FC brakes that itself.
                steady = next((x for x in s if x["t"] > t_trig + 0.3 and tilt(x) < 15
                               and abs(x["vel"][2]) < 3), None)
                if steady is None:
                    reasons.append("never levelled with the climb stopped")
                elif h(at_trigger) > 10 and h(steady) < 5:
                    reasons.append(f"fell to {h(steady):.0f} m before it was steady")
                over = max((tilt(x) for x in s if x["t"] >= t_trig), default=0)
                if over > max(launch_tilt, tilt_now) + 40:
                    reasons.append(f"nose went on to {over:.0f} deg")
            breach = (f"tas {tas:.1f} m/s, tilt {tilt_now:.0f} vs launch {launch_tilt:.0f}"
                      + ("" if not reasons else "; " + "; ".join(reasons)))
            if reasons and ok is not False:
                ok = False
        returning = next((x for x in s if x["stage"] == "RETURN"), None)
        stopped = next((x for x in s if x["t"] >= t_trig
                        and math.hypot(x["vel"][0], x["vel"][1]) <= 2.0
                        and abs(x["vel"][2]) <= 2.0), None)
        stop_s = stop_m = None
        if stopped is not None:
            stop_s = round(stopped["t"] - t_trig, 2)
            stop_m = round(math.hypot(stopped["pos"][0]-at_trigger["pos"][0],
                                      stopped["pos"][1]-at_trigger["pos"][1]), 1)
        return dict(
            name=self.sc.name, ok=ok, expect=self.sc.expect, breach=breach, lifted=self.lifted,
            returned=returned, home_m=None if home_m is None else round(home_m, 1),
            catch=getattr(self, "catch_report", None),
            range_m=round(math.hypot(at_trigger["pos"][0], at_trigger["pos"][1]), 1),
            handover_h=None if hand_h is None else round(hand_h, 1),
            min_h_after=None if min_after is None else round(min_after, 1),
            drop_m=None if hand_h is None else round(hand_h - min_after, 1),
            trigger_h=round(h(at_trigger), 1), trigger_speed=round(vh(at_trigger), 1),
            trigger_tas=round(tas, 1),
            return_vh=None if returning is None else round(vh(returning), 1),
            return_h=None if returning is None else round(h(returning), 1),
            return_tilt=None if returning is None else round(tilt(returning)),
            return_vz=None if returning is None else round(-returning["vel"][2], 1),
            speed_at_mid=None if getattr(self, "speed_at_mid", None) is None else round(self.speed_at_mid, 1),
            stop_s=stop_s, stop_m=stop_m,
            trigger_vz=round(-at_trigger["vel"][2], 1),
            trigger_tilt=round(tilt(at_trigger)), max_tilt=round(max(tilt(x) for x in s)),
            max_rate_dps=round(max(max(abs(r) for r in x["rates"]) for x in self.samples)),
            level_s=None if level_t is None else round(level_t - s[0]["t"], 2),
            max_tilt_after_level=round(max((tilt(x) for x in after_level), default=0)),
            min_h=round(min(h(x) for x in s if not x.get("on_ground")) if any(not x.get("on_ground") for x in s) else 0, 1),
            max_h=round(max(h(x) for x in s), 1),
            descend_s=None if descend is None else round(descend["t"] - s[0]["t"], 1),
            descend_h=None if descend is None else round(h(descend), 1),
            descend_speed=None if descend is None else round(vh(descend), 1),
            yaw_drift_deg=None if yaw_hold is None else round(yaw_hold),
            touchdown_s=None if touch is None else round(touch["t"] - s[0]["t"], 1),
            touchdown_vz=round(final.get("max_td_vz") or 0, 2) if touch else None,
            touchdown_vh=round(final.get("max_td_vh") or 0, 2) if touch else None,
            hard_landing=bool(final.get("hard_landings")) if final else None,
            drift_m=round(math.hypot(final["pos"][0]-at_trigger["pos"][0], final["pos"][1]-at_trigger["pos"][1]), 1) if final else None,
            crashed=crashed, crash_reason=next((x.get("crash_reason") for x in s if x.get("crashed")), None),
            tipped=tipped, disarmed=bool(final) and not final["armed"],
            fallbacks=sorted({x["fallback"] for x in s if x["fallback"]}),
            modes=_collapse([x["mode"] for x in s]),
            events=[e for e in events if any(k in e for k in ("SAFEDOWN", "PILOT", "JETSON", "CATCH", "FENCE"))][:18],
            series=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()
                     if k in ("t", "height", "tilt", "fc_tilt", "fc_h", "fc_climb", "fc_accel_h", "stage", "mode", "airspeed", "fc_pos_ok")}
                    | dict(vz=round(-x["vel"][2], 2), vh=round(vh(x), 2))
                    for x in s[::10]])


def _collapse(seq):
    out = []
    for v in seq:
        if not out or out[-1] != v:
            out.append(v)
    return out


def _one(name, instance, out_dir):
    sc = next(s for s in scenarios() if s.name == name)
    try:
        result = Run(sc, instance, out_dir).run()
    except Exception as exc:
        result = dict(name=sc.name, ok=False, error=f"{type(exc).__name__}: {exc}",
                      trace=traceback.format_exc()[-1500:])
    return result


def _spawn(name, instance, out_dir):
    import subprocess
    path = Path(out_dir)/f"{name}.json"
    cmd = [sys.executable, __file__, "--one", name, "--instance", str(instance), "--out", str(out_dir)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=700)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return dict(name=name, ok=False, error=f"exit {proc.returncode}", trace=proc.stdout[-1500:])


COLS = ("name", "ok", "home_m", "trigger_h", "trigger_vz", "trigger_speed", "max_tilt", "level_s", "min_h",
        "descend_s", "descend_speed", "touchdown_vz", "touchdown_vh", "drift_m", "fallbacks")


def table(results):
    rows = [[str(r.get(c, "")) if c in r else r.get("error", "")[:60] if c == "fallbacks" else ""
             for c in COLS] for r in results]
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(COLS)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(COLS, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="only scenarios whose name contains one of these (comma list)")
    ap.add_argument("-j", type=int, default=3, help="scenarios in parallel")
    ap.add_argument("--out", default=os.environ.get("WASP_SITL_OUT", tempfile.gettempdir()))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--instance", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out)/f"{a.one}.json").write_text(json.dumps(_one(a.one, a.instance, a.out), default=str))
        return 0
    keys = a.k.split(",")
    chosen = [s for s in scenarios() if any(k in s.name for k in keys)]
    if a.list:
        print("\n".join(s.name for s in chosen))
        return 0
    from sitl_launch import find_binary
    if find_binary() is None:
        print("SKIP: no ArduCopter SITL binary (set WASP_SITL_BIN)")
        return 0
    results = []
    import queue
    from concurrent.futures import ThreadPoolExecutor
    free = queue.Queue()
    for i in range(a.j):
        free.put(i)

    def job(name):
        instance = free.get()
        try:
            return _spawn(name, instance, a.out)
        finally:
            free.put(instance)
    with ThreadPoolExecutor(max_workers=a.j) as pool:
        futures = [pool.submit(job, s.name) for s in chosen]
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            tag = "NO-LAUNCH " if r.get("ok") is None else "PASS " if r["ok"] else "FAIL "
            print(tag + r["name"] + ("" if r.get("ok") is not False else f"  {r.get('error', '')}"), flush=True)
    results.sort(key=lambda r: [s.name for s in chosen].index(r["name"]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out)/"sitl_safedown_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(table(results))
    flew = [r for r in results if r.get("ok") is not None]
    print(f"SafeDown: {sum(bool(r['ok']) for r in flew)}/{len(flew)} passed; "
          f"{len(results)-len(flew)} never left the stand (launch findings)")
    return 0 if all(r["ok"] for r in flew) else 1


if __name__ == "__main__":
    sys.exit(main())
