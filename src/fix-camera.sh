#!/usr/bin/env bash
# fix-camera.sh — restore the CSI camera by fixing the DEFAULT BOOT ENTRY.
#
#     sudo ~/Documents/Interceptor-1/src/fix-camera.sh          # fix, then tell you to reboot
#     sudo ~/Documents/Interceptor-1/src/fix-camera.sh --reboot  # fix and reboot immediately
#
# WHAT ACTUALLY BROKE (2026-08-26).  A kernel update (5.15.185 -> 5.15.199) ran
# nv-update-extlinux at 11:32 and rewrote /boot/extlinux/extlinux.conf, flipping
#
#     DEFAULT JetsonIO   ->   DEFAULT primary
#
# The `primary` entry carries NO camera device-tree overlay.  The `JetsonIO`
# entry is the one with
#
#     OVERLAYS /boot/tegra234-p3767-camera-p3768-imx219-A.dtbo
#
# so booting `primary` means the IMX219 is never declared in the device tree,
# nothing binds the sensor, and /dev/video0 cannot appear NO MATTER WHICH CAMERA
# IS PLUGGED IN.  Swapping cameras cannot fix this; neither can modprobe, because
# with no device-tree node there is no I2C device for the driver to bind to.
#
# Proof it was the update and not the hardware: the pre-update backup
# /boot/extlinux/extlinux.conf.nv-update-extlinux-backup still says DEFAULT JetsonIO.
#
# EXPECT THIS TO RECUR ON THE NEXT KERNEL UPDATE.  If the camera ever vanishes
# again after an apt upgrade, this is the first thing to check.
#
# SECOND FAULT, same file, different symptom (2026-08-26, found after the above
# was fixed).  The JetsonIO entry came back pointing at the DUAL overlay:
#
#     OVERLAYS /boot/tegra234-p3767-camera-p3768-imx219-dual.dtbo
#
# That declares TWO IMX219s.  Only one is fitted, so /dev/video0 appears and
# v4l2-ctl lists a sensor — it LOOKS healthy — but Argus is building a two-sensor
# configuration and every capture dies with:
#
#     NvBufSurfaceFromFd Failed          (and no frames, ever)
#
# The tell is that there is no /dev/video1, and the sensor sits on i2c bus 10
# instead of 9.  A minimal `gst-launch-1.0 nvarguscamerasrc ! fakesink` fails
# identically, which is how you know it is not the tracker.  It survives a
# reboot, which is how you know it is not nvargus-daemon state (that is
# restart-camera.sh's job, and it will NOT fix this).
#
# So this script now pins the SINGLE-camera overlay as well as the boot default.
set -u

CONF=/boot/extlinux/extlinux.conf
WANT=JetsonIO

# The module is on CSI port A (verified working: sensor binds at 9-0010 and
# captures frames).  Pass --port-c if it is moved to the other connector.
OVL_DIR=/boot
WANT_OVERLAY="$OVL_DIR/tegra234-p3767-camera-p3768-imx219-A.dtbo"
WANT_DESC="IMX219-A (single, port A)"
for arg in "$@"; do
    case "$arg" in
        --port-c)
            WANT_OVERLAY="$OVL_DIR/tegra234-p3767-camera-p3768-imx219-C.dtbo"
            WANT_DESC="IMX219-C (single, port C)"
            ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "This edits $CONF, so it needs root.  Run:"
    echo "    sudo $0 ${1:-}"
    exit 1
fi

say() { echo; echo "== $* =="; }

say "1. current boot default"
grep -E "^DEFAULT" "$CONF" | sed 's/^/   /'

if ! grep -qE "^LABEL[[:space:]]+$WANT" "$CONF"; then
    echo
    echo "   [FAIL] there is no 'LABEL $WANT' entry in $CONF."
    echo "   The camera entry is gone entirely, not just deselected.  Rebuild it:"
    echo "       sudo /opt/nvidia/jetson-io/jetson-io.py"
    echo "   and choose:  Configure for compatible hardware -> CSI Camera IMX219-A"
    exit 2
fi

# The JetsonIO entry is useless if the files it points at did not survive the
# kernel update, and booting it would then fail rather than just miss the camera.
say "2. checking the files that entry depends on"
ok=1
for f in $(sed -n "/^LABEL[[:space:]]\+$WANT/,/^$/p" "$CONF" \
           | grep -E "^[[:space:]]*(FDT|OVERLAYS)" | awk '{print $2}'); do
    if [ -e "$f" ]; then
        echo "   [ok]   $f"
    else
        echo "   [MISSING] $f"
        ok=0
    fi
done
if [ "$ok" -ne 1 ]; then
    echo
    echo "   [FAIL] that entry points at files that no longer exist — booting it"
    echo "   would fail outright.  Regenerate with:"
    echo "       sudo /opt/nvidia/jetson-io/jetson-io.py"
    exit 3
fi

say "3. camera overlay — must be ONE camera, not the dual"
cur_ovl=$(sed -n "/^LABEL[[:space:]]\+$WANT/,\$p" "$CONF" \
          | grep -E "^[[:space:]]*OVERLAYS" | head -1 | awk '{print $2}')
echo "   want   : $WANT_OVERLAY   ($WANT_DESC)"
echo "   current: ${cur_ovl:-<none>}"

if [ ! -e "$WANT_OVERLAY" ]; then
    echo
    echo "   [FAIL] $WANT_OVERLAY does not exist.  Rebuild the entry with:"
    echo "       sudo /opt/nvidia/jetson-io/jetson-io.py"
    exit 4
fi

if [ "$cur_ovl" = "$WANT_OVERLAY" ]; then
    echo "   already the single-camera overlay — no change needed"
else
    cp -a "$CONF" "$CONF.before-fix-camera.$(date +%Y%m%d-%H%M%S)"
    # Restrict the edit to the JetsonIO block, which is the last one in the file.
    sed -i -E "/^LABEL[[:space:]]+$WANT/,\$ s|^([[:space:]]*OVERLAYS[[:space:]]+).*|\1$WANT_OVERLAY|" "$CONF"
    sed -i -E "/^LABEL[[:space:]]+$WANT/,\$ s|^([[:space:]]*MENU LABEL[[:space:]]+).*|\1Custom Header Config: <CSI Camera $WANT_DESC>|" "$CONF"
    echo "   changed to the single-camera overlay."
fi
sed -n "/^LABEL[[:space:]]\+$WANT/,\$p" "$CONF" | grep -E "^[[:space:]]*(MENU LABEL|OVERLAYS)" | sed 's/^/   now: /'

say "4. setting DEFAULT $WANT"
if grep -qE "^DEFAULT[[:space:]]+$WANT[[:space:]]*$" "$CONF"; then
    echo "   already DEFAULT $WANT — no change needed"
else
    cp -a "$CONF" "$CONF.before-fix-camera.$(date +%Y%m%d-%H%M%S)"
    sed -i -E "s/^DEFAULT[[:space:]]+.*/DEFAULT $WANT/" "$CONF"
    echo "   changed.  Backup kept next to it:"
    ls -1t "$CONF".before-fix-camera.* 2>/dev/null | head -1 | sed 's/^/     /'
fi
grep -E "^DEFAULT" "$CONF" | sed 's/^/   now: /'

say "5. what happens next"
cat <<TXT
   The camera CANNOT come back without a reboot: the device-tree overlay is
   applied by the bootloader, so a running kernel can never gain the node.

   After the reboot, verify in this order:
       ls -l /dev/video0                 # must exist
       v4l2-ctl --list-devices           # must list  vi-output, imx219 9-0010
       systemctl --user is-active wasp-tracker
       tail -f ~/wasp-logs/wasp-tracker.log

   The tracker is already enabled at boot, so if the camera comes up it starts
   following on its own — nothing to type.
TXT

if [ "${1:-}" = "--reboot" ]; then
    say "6. rebooting now"
    sync
    reboot
else
    echo
    echo "=================================================================="
    echo "  Ready.  REBOOT to pick it up:    sudo reboot"
    echo "=================================================================="
fi
