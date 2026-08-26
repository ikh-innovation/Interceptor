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

# WAIT FOR THE CAMERA.  run.py's preflight exits if /dev/video0 is absent, and
# on a cold boot the IMX219 CSI node enumerates on its own schedule — measured
# LATER than this unit's 15 s ExecStartPre.  Without this wait the first few
# boot attempts die on "camera not found" and systemd restarts them every 10 s,
# which reads in the log like a camera fault when it is only a race.
#
# It does NOT paper over a real fault: if the sensor never appears we fall
# through to run.py anyway, so the explicit [PREFLIGHT FAIL] still lands in the
# log.  A sensor that never shows up is a CABLE problem — the IMX219 answers on
# I2C bus 9 at 0x10, and a failed probe logs "error during i2c read probe
# (-121)" to the kernel.  Reseat the CSI ribbon at BOTH ends and reboot.
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
    echo "[boot] the imx219 sensor did not probe — this is almost always the CSI"
    echo "[boot] ribbon cable.  Check:  ls /sys/class/video4linux/   (empty = no sensor)"
fi

# run.py waits for the flight controller to answer a heartbeat, so the Jetson
# may be powered up long before the drone.  It discovers the serial port itself
# (both the by-id descriptor and the ACM number change across reboots).
exec ./run.py
