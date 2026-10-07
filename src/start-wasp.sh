#!/usr/bin/env bash

set -u
cd /home/iknowhow/Documents/Interceptor-1/src || exit 1

export PATH=/home/iknowhow/Documents/Interceptor-1/.env/yolo/bin:$PATH

CONFIG_ENV="$(python3 flight_config.py --shell)" || {
    echo "[boot] flight_config.py refused — see the error above.  Not starting."
    exit 1
}
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

export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

export TRACKER_HEADLESS=${TRACKER_HEADLESS:-1}

export YOLO_AUTOINSTALL=false

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

exec ./run.py
