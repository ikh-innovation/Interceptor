# LTE USB Stick Setup (Alcatel IK41VE1) on Jetson

Gets mobile internet working through an Alcatel IK41VE1 USB broadband stick on
a Jetson running the stock NVIDIA kernel. Tested on L4T R36.5.2, kernel
`5.15.199-tegra`, with a Cosmote SIM (APN `internet`).

This guide is written so **Claude Code can run it step by step**. A human can
follow it the same way.

---

## Instructions for Claude Code

1. Run the steps **in order**. Each step has a **Check** block and a **Pass if**
   condition. Do not continue to the next step until the check passes.
2. If a check fails, look in the [Troubleshooting](#troubleshooting) table. If
   the fix there does not work, **stop and report** the command output to the
   user. Do not improvise other fixes. In particular:
   - do **not** switch the stick to RNDIS/ECM router mode (`AT+USBMODE=1` or `2`).
     On this firmware it has no web UI, never passes traffic, and creates a
     192.168.1.x network that clashes with Wi-Fi;
   - do **not** try to rebuild or replace the kernel.
3. **sudo:** first run `sudo -n true`.
   - If it succeeds, run the sudo commands yourself.
   - If it fails (password required), do **not** ask the user to type their
     password into the chat. Print the exact sudo block for the step and ask the
     user to run it in their own terminal, then run the step's **Check**
     yourself.
   - If a sudo command is blocked by Claude Code's permission system, hand
     that block to the user in the same way.
4. Run everything non-interactively. Do not use `minicom`, `screen` or other
   interactive tools. This guide provides a small Python helper for AT commands.
5. Steps that change networking (Step 7, Step 8) change which network the
   Jetson uses for internet. Ask the user before running Step 8.
6. All commands use `$HOME/lte-kmod` as the work folder.

### Why this is needed (background)

The stock Jetson kernel ships **without** PPP and MBIM drivers
(`# CONFIG_PPP is not set`, `# CONFIG_USB_NET_CDC_MBIM is not set`).
ModemManager detects the stick and registers on the network, but connecting
fails with `PPP failed` / `Couldn't open the /dev/ppp device`. The fix is to
build the missing kernel modules out-of-tree against the installed kernel
headers. After that NetworkManager dials the stick over PPP on `ttyUSB0`.

---

## Step 1 — Is it already working?

**Check**
```bash
ip -br addr show ppp0 2>/dev/null; ping -c2 -W3 -I ppp0 8.8.8.8 2>/dev/null | tail -1
```

**Pass if** `ppp0` has an address and the ping shows `0% packet loss`.
If it passes, the stick already works. Go straight to Step 8 (optional) and stop.
Otherwise continue.

## Step 2 — Stick detected, SIM registered

The stick must be plugged in with the SIM inserted for about 20 s.

**Check**
```bash
lsusb | grep 1bbb
mmcli -L
M=$(mmcli -L | grep -o 'Modem/[0-9]*' | cut -d/ -f2); echo "modem=$M"
[ -n "$M" ] && mmcli -m "$M" | grep -E " state:|signal quality|operator name|primary port"
```

**Pass if** all of these hold:
- `lsusb` shows `1bbb:00b6`;
- `mmcli -L` lists `[Alcatel] IK41VE1`;
- `state:` is `registered` or `connected`.

If it fails:
- `lsusb` shows `1bbb:01aa`: the stick is in router mode. Do "Reset stick to mode 0" in Troubleshooting.
- `state: locked`: the SIM needs its PIN. Ask the user for it, then run `mmcli -i 0 --pin=XXXX`.
- `state: enabling` or `searching`: wait 30 s and check again.

## Step 3 — Prerequisites

**Check**
```bash
uname -r
ls /lib/modules/$(uname -r)/build/Module.symvers
which gcc make curl
zcat /proc/config.gz | grep -E "CONFIG_PPP=|CONFIG_PPP is not set|CONFIG_MODULE_SIG_FORCE"
ls /lib/modules/$(uname -r)/updates/wwan/ 2>/dev/null
```

**Pass if**:
- `Module.symvers` exists;
- `gcc`, `make` and `curl` are all found;
- `CONFIG_MODULE_SIG_FORCE` is **not** `=y`.

Interpreting the output:
- If the output shows `CONFIG_PPP=y` or `=m`, the kernel already has PPP. Skip Steps 4–6 and go to Step 7.
- If `updates/wwan/` already lists the 5 `.ko` files, the modules are already
  built for this kernel. Skip to Step 6.
- If `Module.symvers` is missing, ask the user to run `sudo apt install -y nvidia-l4t-kernel-headers`, then recheck.

## Step 4 — Download the driver sources

The sources must match the running kernel version: `5.15.199-tegra` uses tag `v5.15.199`.

```bash
KVER=$(uname -r | cut -d- -f1)
mkdir -p "$HOME/lte-kmod" && cd "$HOME/lte-kmod"
B=https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/plain
for f in drivers/net/ppp/ppp_generic.c \
         drivers/net/ppp/ppp_async.c \
         drivers/net/slip/slhc.c \
         drivers/net/usb/cdc_mbim.c \
         drivers/usb/class/cdc-wdm.c; do
  curl -sf -o "$(basename $f)" "$B/$f?h=v$KVER" && echo "ok   $f" || echo "FAIL $f"
done
```

**Pass if** all 5 lines say `ok`.
If any line says `FAIL`, the tag doesn't exist. Report `uname -r` to the user and stop.

## Step 5 — Build the modules

```bash
cd "$HOME/lte-kmod"
echo 'obj-m := ppp_generic.o ppp_async.o slhc.o cdc_mbim.o cdc-wdm.o' > Makefile
make -C /lib/modules/$(uname -r)/build M="$PWD" modules -j$(nproc) 2>&1 | grep -E "ERROR|error:|warning: the compiler"
ls *.ko
```

**Pass if** `ls` shows exactly these 5 files: `cdc_mbim.ko cdc-wdm.ko ppp_async.ko ppp_generic.ko slhc.ko`.
The warning `the compiler differs from the one used to build the kernel` is
harmless. Any `ERROR` line counts as a failure: stop and report it.

## Step 6 — Install and load the modules (sudo)

**sudo block**
```bash
cd "$HOME/lte-kmod"
sudo mkdir -p /lib/modules/$(uname -r)/updates/wwan
sudo cp *.ko /lib/modules/$(uname -r)/updates/wwan/
sudo depmod -a
sudo modprobe ppp_generic
sudo modprobe ppp_async
sudo modprobe cdc_mbim
printf 'ppp_generic\nppp_async\n' | sudo tee /etc/modules-load.d/ppp.conf
sudo systemctl restart ModemManager
```

> Use one `modprobe` per module. `modprobe a b` does **not** load two modules;
> it passes `b` as an option to `a`.

**Check** (wait about 20 s after the ModemManager restart)
```bash
ls -l /dev/ppp
lsmod | grep -E "^(ppp_generic|ppp_async|slhc) "
timeout 60 bash -c 'until mmcli -L 2>/dev/null | grep -q IK41; do sleep 3; done'; mmcli -L
```

**Pass if** `/dev/ppp` exists, all 3 modules are listed, and `mmcli -L` shows the IK41VE1.

## Step 7 — Create and start the connection (sudo)

First check whether a GSM profile already exists:

```bash
nmcli -t -f NAME,TYPE con show | grep gsm
```

- **If one exists** (e.g. `Default:gsm`), check its APN with
  `nmcli -g gsm.apn con show Default`. For Cosmote it must be `internet`. For
  any other provider, ask the user for the APN.
- **If none exists**, it needs to be created (first line of the block below).

**sudo block**
```bash
# only if no gsm profile exists:
sudo nmcli con add type gsm ifname '*' con-name Default apn internet
# always:
sudo nmcli con mod Default connection.autoconnect yes
sudo nmcli con up Default
```

**Check**
```bash
ip -br addr show ppp0
ping -c3 -W3 -I ppp0 8.8.8.8 | tail -2
curl -s -m 10 --interface ppp0 https://ifconfig.me; echo
```

**Pass if** `ppp0` has an address (usually `10.x.x.x`), the ping shows `0% packet loss`, and `curl` prints a public IP.

When this passes, setup is complete. Tell the user the `ppp0` address and the
public IP.

## Step 8 (optional, ask the user first) — Make the stick the main route

By default Wi-Fi/Ethernet stay the main internet route while connected, and
the stick takes over only when they drop. To make the stick the main route
even with Wi-Fi up (Wi-Fi's metric is usually 600):

**sudo block**
```bash
sudo nmcli con mod Default ipv4.route-metric 50
sudo nmcli con up Default
```

**Check**
```bash
ip route | grep default
```

**Pass if** the first `default` line is `dev ppp0`.

Undo with `sudo nmcli con mod Default ipv4.route-metric -1 && sudo nmcli con up Default`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection activation failed: PPP failed` | PPP modules not loaded. Redo the Step 6 check. Check `journalctl -u NetworkManager -n 30` for `/dev/ppp`. |
| `modprobe: FATAL: Module ... not found` | Modules not installed for the running kernel. Redo Steps 4–6. |
| `Invalid module format` / `disagrees about version` | Sources don't match `uname -r`. Redo Steps 4–6. |
| `mmcli -L` shows nothing | Wait 30 s. Run `sudo systemctl restart ModemManager`. Replug the stick. |
| `lsusb` shows `1bbb:01aa`, or a new `usbN` interface with a `192.168.1.x` address | Stick is in router mode. Do "Reset stick to mode 0" below. |
| Connects but no internet | Wrong APN. Check `nmcli -g gsm.apn con show Default`. |
| Everything worked before a JetPack/kernel upgrade, now broken | Modules are per kernel. Redo Steps 3–6. |

### AT command helper

AT commands need ModemManager stopped, because it holds the serial port.
Create the helper once:

```bash
mkdir -p "$HOME/lte-kmod"
cat > "$HOME/lte-kmod/at.py" <<'EOF'
import os, sys, termios, time, select, tty
fd = os.open(sys.argv[1], os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
tty.setraw(fd)
a = termios.tcgetattr(fd); a[4] = a[5] = termios.B115200; a[2] |= termios.CLOCAL | termios.CREAD
termios.tcsetattr(fd, termios.TCSANOW, a)
for cmd in sys.argv[2:]:
    os.write(fd, (cmd + "\r").encode())
    out = b""; end = time.time() + 2.5
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.2)
        if r: out += os.read(fd, 4096)
    print(">>>", cmd, "\n", out.decode(errors="replace").strip(), "\n")
os.close(fd)
EOF
```

Usage (read-only queries):
```bash
sudo systemctl stop ModemManager
python3 -I "$HOME/lte-kmod/at.py" /dev/ttyUSB0 ATI 'AT+USBMODE?' 'AT+COPS?'
sudo systemctl start ModemManager
```

What the queries mean:
- `AT+USBMODE?` shows the current mode. It must be `0`.
- `AT+USBMODE=?` lists the modes: `0` modem+diag+at_mbim, `1` rndis, `2` ecm.
- `AT+COPS?` shows the registered operator.

The user must be in the `dialout` group to open `/dev/ttyUSB*`. Check with `id`.

### Reset stick to mode 0 (only if it is in router mode, `1bbb:01aa`)

**sudo block**
```bash
sudo systemctl stop ModemManager
echo "1bbb 01aa" | sudo tee /sys/bus/usb-serial/drivers/option1/new_id
```

Then:
```bash
python3 -I "$HOME/lte-kmod/at.py" /dev/ttyUSB0 'AT+USBMODE=0'
timeout 60 bash -c 'until lsusb | grep -q 1bbb:00b6; do sleep 2; done'; lsusb | grep 1bbb
sudo systemctl start ModemManager
```

**Pass if** `lsusb` shows `1bbb:00b6`. Then go back to Step 2.
If `/dev/ttyUSB0` does not appear after the `new_id` command, try `/dev/ttyUSB1`.

## Notes

- The MBIM driver (`cdc_mbim`) is built, but the `option` serial driver claims
  the stick's data interface. The working, tested path is **PPP over
  `ttyUSB0`**. It is fine for telemetry and browsing.
- Modules in `/lib/modules/<kernel>/updates/` survive reboots but **not**
  kernel upgrades.
- Auto-connect is enabled in Step 7, so the stick reconnects by itself after a
  reboot or replug.
