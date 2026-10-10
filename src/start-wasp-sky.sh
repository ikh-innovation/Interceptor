#!/usr/bin/env bash

set -u
WASP_SRC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
cd "$WASP_SRC_DIR" || exit 1

export PATH="$WASP_SRC_DIR/../.env/yolo/bin:$PATH"

export CAM_TILT_DEG=${CAM_TILT_DEG:-30}

export TARGET_WIDTH_M=${TARGET_WIDTH_M:-0.30}

export COMMIT_CHAN=${COMMIT_CHAN:-0}

export MAX_LEAN_DEG=${MAX_LEAN_DEG:-12}
export ROLL_PN_GAIN=${ROLL_PN_GAIN:-35}

export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

export YOLO_AUTOINSTALL=false

if systemctl --user is-active --quiet wasp-tracker; then
    systemctl --user stop wasp-tracker
    for _ in $(seq 20); do
        systemctl --user is-active --quiet wasp-tracker || break
        sleep 0.5
    done
fi

if [ "${SKY_TILT_OK:-0}" != "1" ]; then
    if [ -t 0 ]; then
        read -r -p "Camera mounted ${CAM_TILT_DEG} degrees UP? Type yes: " _ans
        case "${_ans}" in
            yes|YES|y|Y) ;;
            *) exit 1 ;;
        esac
    else
        exit 1
    fi
fi

CAM_WAIT_S=${CAM_WAIT_S:-90}
if [ ! -e /dev/video0 ]; then
    for _ in $(seq "$CAM_WAIT_S"); do
        [ -e /dev/video0 ] && break
        sleep 1
    done
fi
exec ./run.py
