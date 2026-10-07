#!/usr/bin/env bash

set -u
WASP_SRC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$WASP_SRC_DIR" || exit 1

export PATH="$WASP_SRC_DIR/../.env/yolo/bin:$PATH"

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
echo "[boot] normal guidance tilt ${CAM_TILT_DEG} deg; physical camera mount ${CAMERA_MOUNT_DEG} deg"
echo "[boot] choose Normal or Wasp on the phone; target width for normal guidance ${TARGET_WIDTH_M} m"
echo "[boot] visible band ${WASP_MODE_BAND}"

export TRACKER_HEADLESS=${TRACKER_HEADLESS:-1}
if [ "$TRACKER_HEADLESS" != "1" ]; then
    export DISPLAY=${DISPLAY:-:0}
    export XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
fi

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
