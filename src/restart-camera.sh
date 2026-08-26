#!/usr/bin/env bash
# restart-camera.sh — recover the CSI camera at RUNTIME and start the tracker.
#
#     sudo ~/Documents/Interceptor-1/src/restart-camera.sh
#
# USE THIS WHEN the camera enumerates (/dev/video0 exists, v4l2-ctl lists the
# sensor) but every capture fails with:
#
#     NvBufSurfaceFromFd Failed
#     nvbuf_utils: dmabuf_fd -1 mapped entry NOT found
#
# That is nvargus-daemon's buffer state, not the cable, not the overlay and not
# the pipeline — a minimal `gst-launch-1.0 nvarguscamerasrc ! fakesink` fails
# identically.  It is provoked by repeatedly opening the camera with parameters
# the sensor cannot satisfy, which is exactly what the tracker's 10 s restart
# loop does when the sensor mode is wrong for the fitted module.  The daemon does
# not recover on its own and stopping every client does not clear it.
#
# NOT for a missing /dev/video0 — that is the boot overlay.  Use fix-camera.sh.
set -u

if [ "$(id -u)" -ne 0 ]; then
    echo "This restarts a system daemon, so it needs root.  Run:"
    echo "    sudo $0"
    exit 1
fi

USER_NAME=iknowhow
UID_NUM=$(id -u "$USER_NAME")
as_user() {
    sudo -u "$USER_NAME" XDG_RUNTIME_DIR="/run/user/$UID_NUM" "$@"
}

say() { echo; echo "== $* =="; }

say "1. is the sensor even enumerated?"
if [ ! -e /dev/video0 ]; then
    echo "   /dev/video0 is ABSENT — this is the wrong tool."
    echo "   That is the boot-time device-tree overlay.  Use:"
    echo "       sudo $(dirname "$0")/fix-camera.sh"
    exit 2
fi
echo "   $(cat /sys/class/video4linux/video0/name 2>/dev/null || echo '/dev/video0')"

say "2. stopping the tracker so it stops re-poisoning the daemon"
as_user systemctl --user stop wasp-tracker.service 2>/dev/null
sleep 2
echo "   stopped"

say "3. restarting nvargus-daemon"
systemctl restart nvargus-daemon
sleep 3
systemctl is-active nvargus-daemon | sed 's/^/   nvargus-daemon: /'

say "4. proving capture works again (30 real frames)"
if as_user timeout 60 gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=30 \
        ! fakesink >/tmp/rc.$$ 2>&1 && ! grep -q "NvBufSurfaceFromFd Failed" /tmp/rc.$$; then
    echo "   OK — the camera streams"
    rm -f /tmp/rc.$$
else
    echo "   STILL FAILING:"
    grep -iE "error|NvBuf|dmabuf" /tmp/rc.$$ | head -6 | sed 's/^/     /'
    rm -f /tmp/rc.$$
    echo
    echo "   The daemon restart did not clear it.  REBOOT — that always does:"
    echo "       sudo reboot"
    echo "   The tracker is enabled at boot, so it comes back up on its own."
    exit 3
fi

say "5. PRESS RUN"
: > /home/iknowhow/wasp-logs/wasp-tracker.log
as_user systemctl --user reset-failed wasp-tracker.service 2>/dev/null
as_user systemctl --user start wasp-tracker.service
echo "   started — waiting for the first frames..."
sleep 60

as_user systemctl --user is-active wasp-tracker.service | sed 's/^/   service: /'
echo "   restarts: $(as_user systemctl --user show wasp-tracker -p NRestarts --value)"
echo
echo "=================================================================="
grep -E "\[boot\] sensor|\[RUN\]|\[REC\] this flight|PREFLIGHT|no frames" \
     /home/iknowhow/wasp-logs/wasp-tracker.log 2>/dev/null | head -6
echo "---"
grep "CTRL" /home/iknowhow/wasp-logs/wasp-tracker.log 2>/dev/null | tail -2 | cut -c1-150
echo "=================================================================="
echo
echo "A [CTRL] line means it is running.  Watch it live:"
echo "    tail -f ~/wasp-logs/wasp-tracker.log"
