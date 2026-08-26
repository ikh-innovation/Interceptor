#!/usr/bin/env bash
# WASP head tracker — power-on launcher.
#
# This is the ONE file to edit in the field.  It is what the systemd user unit
# runs at Jetson power-up, and it is also runnable by hand:
#
#     ~/Documents/Interceptor-1/src/start-wasp.sh
#
# Autostart:  systemctl --user enable --now wasp-tracker.service
# Watch:      tail -f ~/wasp-logs/wasp-tracker.log
# Stop:       systemctl --user stop wasp-tracker     (releases every channel)

set -u
cd /home/iknowhow/Documents/Interceptor-1/src || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# 1. CAMERA TILT, IN DEGREES UP.  A HARDWARE MEASUREMENT — measure the mount.
#    run.py refuses to start without it, on purpose.  The control law adds this
#    to every elevation it computes, so a wrong number flies the airframe high
#    or low by exactly that error, consistently, with nothing in the log to say
#    why.  10 is the recommended mount, NOT a safe default for an untilted lens.
export CAM_TILT_DEG=10

# 2. COMMIT / ABORT CHANNEL.  A spare RC channel: scales the lean AND aborts in
#    ~100 ms.  Leave EMPTY if no spare channel is wired — with it set but not
#    wired, commit reads 0 and the tracker deliberately refuses to steer.
#    MUST NOT be the flight-mode channel (this FC has FLTMODE_CH=6), and must
#    not be 1-4 (those are the driven sticks).  So: 5, 7 or 8.
export COMMIT_CHAN=${COMMIT_CHAN:-0}

# 3. Gentler first flight.  Comment both out for the full shipped 20 deg / 45.
#    MAX_LEAN_DEG and ROLL_PN_GAIN must ALWAYS move together (gain ~= 3 x speed).
export MAX_LEAN_DEG=12
export ROLL_PN_GAIN=35
# ─────────────────────────────────────────────────────────────────────────────

# The cv2 HUD needs the autologin session's X display.
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

# THE FIELD HAS NO WIFI.  The detector is pulled from the HuggingFace cache;
# without this, huggingface_hub can stall on a DNS/HTTP timeout at every boot.
# Verified: the cached model loads with HF_HUB_OFFLINE=1.
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# run.py waits for the flight controller to answer a heartbeat, so the Jetson
# may be powered up long before the drone.  It discovers the serial port itself
# (both the by-id descriptor and the ACM number change across reboots).
exec ./run.py
