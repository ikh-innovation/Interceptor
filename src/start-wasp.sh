#!/usr/bin/env bash

set -u
WASP_SRC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$WASP_SRC_DIR" || exit 1

export PATH="$WASP_SRC_DIR/../.env/yolo/bin:$PATH"

CONFIG_ENV="$(python3 flight_config.py --shell)" || {
    exit 1
}
while IFS='=' read -r _k _v; do
    [ -z "${_k}" ] && continue
    eval "current=\${${_k}:-}"
    if [ -n "${current}" ]; then
        :
    else
        eval "export ${_k}=${_v}"
    fi
done <<< "${CONFIG_ENV}"

export TRACKER_HEADLESS=${TRACKER_HEADLESS:-1}
if [ "$TRACKER_HEADLESS" != "1" ]; then
    export DISPLAY=${DISPLAY:-:0}
    export XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}
fi

export YOLO_AUTOINSTALL=false

CAM_WAIT_S=${CAM_WAIT_S:-90}
if [ ! -e /dev/video0 ]; then
    for _ in $(seq "$CAM_WAIT_S"); do
        [ -e /dev/video0 ] && break
        sleep 1
    done
fi
exec ./run.py
