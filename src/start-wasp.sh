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
# WHICH MODE FLIES IS DECIDED IN flight_config.py, NOT HERE.
#
# That file is the one place to edit: pick MODE = "human" or "sky", declare the
# real MOUNT_TILT_DEG, reboot.  It refuses to start if the mode's required
# camera angle disagrees with the mount you declared, because a wrong tilt does
# not error — it flies high or low by exactly that error, permanently.
#
# Everything below (camera tilt, target width, lean, gain, commit channel) is
# whatever flight_config.py resolved.  To override for ONE run without editing
# it, export the variable before calling this script:
#
#     MAX_LEAN_DEG=16 ROLL_PN_GAIN=48 ./start-wasp.sh
#
CONFIG_ENV="$(python3 flight_config.py --shell)" || {
    echo "[boot] flight_config.py refused — see the error above.  Not starting."
    exit 1
}
# Pre-set variables win, so a one-off export still overrides the file.
while IFS='=' read -r _k _v; do
    [ -z "${_k}" ] && continue
    eval "current=\${${_k}:-}"
    if [ -n "${current}" ]; then
        echo "[boot] ${_k} overridden from the environment: ${current}"
    else
        eval "export ${_k}=${_v}"
    fi
done <<< "${CONFIG_ENV}"

echo "[boot] mode: ${WASP_MODE} — ${WASP_MODE_LABEL}"
echo "[boot] camera tilt ${CAM_TILT_DEG} deg, target width ${TARGET_WIDTH_M} m"
echo "[boot] visible band ${WASP_MODE_BAND}"
# ─────────────────────────────────────────────────────────────────────────────

# The cv2 HUD needs the autologin session's X display.
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

# HEADLESS IN THE FIELD.  The local cv2 window has a 'q' shortcut that quits the
# tracker, and quitting is a CLEAN exit, which Restart=on-failure deliberately
# does not undo — so one stray keypress at the Jetson takes the seeker down and
# leaves it down.  Observed repeatedly on 2026-09-14.  The phone dashboard now
# shows everything the window did, so the window is pure risk during a flight;
# it also costs ~1.8 ms/frame of the 33 ms budget.
#   TRACKER_HEADLESS=0 ./start-wasp.sh    brings the local window back for bench work
export TRACKER_HEADLESS=${TRACKER_HEADLESS:-1}

# THE FIELD HAS NO WIFI.  The detector is pulled from the HuggingFace cache;
# without this, huggingface_hub can stall on a DNS/HTTP timeout at every boot.
# Verified: the cached model loads with HF_HUB_OFFLINE=1.
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# ULTRALYTICS ALSO REACHES FOR THE NETWORK, and HF_HUB_OFFLINE does not stop it.
# On first load it decided a requirement was missing and ran pip at RUNTIME
# ("requirements: AutoUpdate success") — on the bench that is a 7 s pause, in
# the field it is a pip call with no route, inside the boot path, before the
# first frame.  Off.  Missing packages are an install-time problem, and the
# preflight is where they should surface.
export YOLO_AUTOINSTALL=false

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
