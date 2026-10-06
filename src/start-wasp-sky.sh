#!/usr/bin/env bash
# WASP — SKY TARGET launcher, by hand, with a mount-tilt confirmation.
#
# The boot default is start-wasp.sh, which flies the same sky mode from
# flight_config.py and is what the systemd unit runs at power-up.  This script
# is the manual way in: it asks you to confirm the camera is physically at
# 30 deg before anything starts.
#
#     ~/Documents/Interceptor-1/src/start-wasp-sky.sh
#
# Back to the boot service:  systemctl --user start wasp-tracker
# Watch the boot service:    tail -f ~/wasp-logs/wasp-tracker.log
# Stop this one:             Ctrl-C, or q in the window
#
# Measured in the closed-loop sim (tests/test_intercept.py), target hovering
# above the launch point: camera tilt 30, climb 6.0 m/s closes to ~35 deg.
#
# Both the 30 deg tilt and the faster climb are needed: at tilt 30 with the
# stock 2.5 m/s climb the target is 94% in frame and it STILL misses by 5.9 m,
# because the airframe flies forward at ~11 m/s and can only climb at 2.25 —
# an 11 deg climb angle.  It goes underneath.
#
# Above ~45 deg nothing closes, in any configuration tried.  The pitch channel
# commands lean from AZIMUTH only (fc_control.py, "PITCH — closing speed"):
# there is no elevation term, so it cannot trade forward speed for climb and a
# near-vertical intercept is not in this law.  Do not expect one.

set -u
cd /home/iknowhow/Documents/Interceptor-1/src || exit 1

# Same venv as start-wasp.sh — see the note there.
export PATH=/home/iknowhow/Documents/Interceptor-1/.env/yolo/bin:$PATH

# ─────────────────────────────────────────────────────────────────────────────
# 1. CAMERA TILT — A HARDWARE FACT, NOT A PREFERENCE.
#    This mode is scored at 30 deg.  The control law ADDS this to every
#    elevation it computes, so if the number here and the real mount disagree,
#    the airframe flies high or low by exactly that error, consistently, and
#    NOTHING in the log will say why.  The camera must physically be at 30.
export CAM_TILT_DEG=${CAM_TILT_DEG:-30}

# 2. TARGET WIDTH, METRES — the target's REAL width, not its apparent size.
#    Range is focal_px * this / bbox_px, so it is what turns pixels into metres:
#    it drives the range readout AND the terminal freeze.  MEASURE the drone motor-tip to motor-tip and set it.
#    0.30 is a placeholder for a 7-inch quad, i.e. an assumption.
export TARGET_WIDTH_M=${TARGET_WIDTH_M:-0.30}

# 3. COMMIT / ABORT CHANNEL.  0 = none wired, so lean is
#    pinned at 100% whenever a target is locked and the mode switch is the only
#    abort (~1 s at heartbeat rate).
export COMMIT_CHAN=${COMMIT_CHAN:-0}

# 4. Lean and turn gain, a gentle pair.  These two ALWAYS
#    move together (gain ~= 3 x cruise speed) — raising the lean alone makes it
#    trail.  Left gentle on purpose: this mode has never been flown.
export MAX_LEAN_DEG=${MAX_LEAN_DEG:-12}
export ROLL_PN_GAIN=${ROLL_PN_GAIN:-35}
# ─────────────────────────────────────────────────────────────────────────────

# The cv2 HUD needs the autologin session's X display.
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

# No WiFi in the field: ultralytics must not run pip.
export YOLO_AUTOINSTALL=false

echo "=========================================================================="
echo " WASP — SKY TARGET MODE"
echo "=========================================================================="
echo "  camera tilt   : ${CAM_TILT_DEG} deg UP   <-- THE MOUNT MUST BE AT THIS ANGLE"
echo "  target width  : ${TARGET_WIDTH_M} m       <-- the REAL width of the drone"
echo "  lean / gain   : ${MAX_LEAN_DEG} deg / ${ROLL_PN_GAIN}"
echo "  climb ceiling : PILOT_SPEED_UP on the FC (set to 600 = 6 m/s)"
echo "--------------------------------------------------------------------------"
echo "  Scored envelope: closes to ~35 deg elevation.  Above ~45 deg it does"
echo "  NOT close, in any configuration — the pitch channel has no elevation"
echo "  term and cannot climb near-vertically."
echo "=========================================================================="

# THE PORT IS SINGLE-OWNER.  The boot service holds /dev/ttyACM0; starting a
# second run against the same port fails deep inside the firmware probe with
# "device reports readiness to read but returned no data", which reads like a
# cable fault and is not one.  Take the port cleanly instead.
if systemctl --user is-active --quiet wasp-tracker; then
    echo "[sky] the boot service is running and owns the FC port — stopping it."
    systemctl --user stop wasp-tracker
    for _ in $(seq 20); do
        systemctl --user is-active --quiet wasp-tracker || break
        sleep 0.5
    done
    echo "[sky] stopped.  It returns on reboot, or:"
    echo "      systemctl --user start wasp-tracker"
fi

# THE MOUNT CHECK.  This is the one failure that hides: a wrong tilt does not
# error, it just flies wrong forever.  Confirm it out loud when an operator is here.
if [ "${SKY_TILT_OK:-0}" != "1" ]; then
    if [ -t 0 ]; then
        echo
        echo "  Is the camera PHYSICALLY tilted to ${CAM_TILT_DEG} deg UP?"
        read -r -p "  Type yes to continue: " _ans
        case "${_ans}" in
            yes|YES|y|Y) ;;
            *) echo "[sky] aborted — nothing was started."; exit 1 ;;
        esac
    else
        echo "[sky] not a terminal and SKY_TILT_OK is not 1 — refusing to start."
        echo "      Re-run from a terminal, or SKY_TILT_OK=1 $0"
        exit 1
    fi
fi

# WAIT FOR THE CAMERA — the CSI node enumerates on its own schedule after a
# cold boot.  Falls through to run.py either way, so a real fault still lands
# in the preflight as [PREFLIGHT FAIL] rather than being papered over here.
CAM_WAIT_S=${CAM_WAIT_S:-90}
if [ ! -e /dev/video0 ]; then
    echo "[sky] waiting up to ${CAM_WAIT_S}s for /dev/video0 ..."
    for _ in $(seq "$CAM_WAIT_S"); do
        [ -e /dev/video0 ] && break
        sleep 1
    done
fi
[ -e /dev/video0 ] && echo "[sky] camera present: /dev/video0" \
                   || echo "[sky] /dev/video0 ABSENT — check the CSI ribbon."

exec ./run.py
