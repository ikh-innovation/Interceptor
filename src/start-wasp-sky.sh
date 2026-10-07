#!/usr/bin/env bash

set -u
cd /home/iknowhow/Documents/Interceptor-1/src || exit 1

export PATH=/home/iknowhow/Documents/Interceptor-1/.env/yolo/bin:$PATH

export CAM_TILT_DEG=${CAM_TILT_DEG:-30}

export TARGET_WIDTH_M=${TARGET_WIDTH_M:-0.30}

export COMMIT_CHAN=${COMMIT_CHAN:-0}

export MAX_LEAN_DEG=${MAX_LEAN_DEG:-12}
export ROLL_PN_GAIN=${ROLL_PN_GAIN:-35}

export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority

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
