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
# 1. CAMERA TILT, IN DEGREES UP.  Now HARDCODED to 10 inside run.py
#    (HARDCODED_CAM_TILT_DEG), so exactly ONE place knows the mount angle and
#    this script can no longer disagree with it.  Left commented out on purpose:
#    uncomment ONLY to override run.py for a one-off run.
#
#    It is a HARDWARE FACT.  The control law adds it to every elevation it
#    computes, so a wrong number flies the airframe high or low by exactly that
#    error, consistently, with nothing in the log to say why.
# export CAM_TILT_DEG=10

# 2. COMMIT / ABORT CHANNEL.  A spare RC channel: scales the lean AND aborts in
#    ~100 ms.  Leave EMPTY if no spare channel is wired — with it set but not
#    wired, commit reads 0 and the tracker deliberately refuses to steer.
#    MUST NOT be the flight-mode channel (this FC has FLTMODE_CH=6), and must
#    not be 1-4 (those are the driven sticks).  So: 5, 7 or 8.
export COMMIT_CHAN=${COMMIT_CHAN:-0}

# 3. CAMERA — PINNED TO THE IMX219.  This airframe carries one module and only
#    one, so the mode is stated outright instead of branched on.  Asking for a
#    mode the sensor does not have is a HARD failure, not a fallback ("Frame Rate
#    specified is greater than supported" / no frames at all), so the numbers
#    below must stay consistent with each other.
#
#      IMX219 modes: 0:3280x2464@21  1:3280x1848@28  2:1920x1080@30
#                    3:1640x1232@30  4:1280x720@60
#
#    Mode 4 at 30 fps: 1280x720 is native to the sensor, and 30 keeps the
#    detector's per-frame budget where it was measured (~97 ms at imgsz 640).
#    Any of these can still be overridden from the environment for one run.
export CAM_SENSOR_MODE=${CAM_SENSOR_MODE:-4}
export CAM_WIDTH=${CAM_WIDTH:-1280}
export CAM_HEIGHT=${CAM_HEIGHT:-720}
export CAM_FPS=${CAM_FPS:-30}

#    Confirm the sensor actually bound is the one these numbers describe.  This
#    is a WARNING, not a gate: the run continues, because a wrong-but-present
#    sensor still gives a picture to debug with, whereas refusing to start gives
#    nothing.  A MISSING sensor is a different fault with a different fix — the
#    boot device-tree overlay, see fix-camera.sh.
CAM_NAME="$(cat /sys/class/video4linux/video0/name 2>/dev/null || true)"
case "$CAM_NAME" in
    *imx219*) echo "[boot] sensor: IMX219 -> mode 4, 1280x720@30" ;;
    "")       echo "[boot] WARNING: no sensor bound (/dev/video0 has no name)." ;;
    *)        echo "[boot] WARNING: expected an IMX219 but the bound sensor is"
              echo "[boot]          '$CAM_NAME'.  Mode 4 / 1280x720 may not exist"
              echo "[boot]          on it, and the LENS FOV below is the IMX219's." ;;
esac

#    LENS FIELD OF VIEW.  fc_control's default 62.2 x 48.8 deg IS this IMX219
#    module's lens, so nothing needs setting here.  Left documented because FOV
#    rescales the control loop's gain and sets the visible band — swap the lens or
#    the module and these become wrong, silently.  Prefer tuning KP_* over
#    chasing optics; only change these if the glass actually changed.
# export CAM_HFOV_DEG=62.2
# export CAM_VFOV_DEG=48.8

# 4. Gentler first flight.  Comment both out for the full shipped 20 deg / 45.
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

# WAIT FOR THE CAMERA.  run.py's preflight exits if /dev/video0 is absent, and
# on a cold boot the IMX219 CSI node enumerates on its own schedule — measured
# LATER than this unit's 15 s ExecStartPre.  Without this wait the first few
# boot attempts die on "camera not found" and systemd restarts them every 10 s,
# which reads in the log like a camera fault when it is only a race.
#
# It does NOT paper over a real fault: if the sensor never appears we fall
# through to run.py anyway, so the explicit [PREFLIGHT FAIL] still lands in the
# log.
#
# A sensor that NEVER shows up is NOT the cable.  On 2026-08-26 it was the boot
# device-tree overlay: a kernel update rewrote /boot/extlinux/extlinux.conf and
# flipped DEFAULT JetsonIO -> DEFAULT primary, and `primary` declares no camera,
# so nothing can bind no matter which module is plugged in.  Chasing the ribbon
# first cost a bring-up cycle.  Check the cheap thing first:
#     grep ^DEFAULT /boot/extlinux/extlinux.conf     # must say JetsonIO
#     sudo ./fix-camera.sh                           # fixes it, expect it to recur
#                                                    # on the next kernel update
CAM_WAIT_S=${CAM_WAIT_S:-90}
if [ ! -e /dev/video0 ]; then
    echo "[boot] waiting up to ${CAM_WAIT_S}s for /dev/video0 (CSI sensor enumerating)..."
    for _ in $(seq "$CAM_WAIT_S"); do
        [ -e /dev/video0 ] && break
        sleep 1
    done
fi
if [ -e /dev/video0 ]; then
    echo "[boot] camera present: /dev/video0"
else
    echo "[boot] /dev/video0 STILL ABSENT after ${CAM_WAIT_S}s."
    echo "[boot] the imx219 did not bind.  Check the BOOT OVERLAY first, not the cable:"
    echo "[boot]     grep ^DEFAULT /boot/extlinux/extlinux.conf   # must say JetsonIO"
    echo "[boot]     sudo ./fix-camera.sh"
fi

# run.py waits for the flight controller to answer a heartbeat, so the Jetson
# may be powered up long before the drone.  It discovers the serial port itself
# (both the by-id descriptor and the ACM number change across reboots).
exec ./run.py
