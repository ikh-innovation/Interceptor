# Add a new Jetson to the fleet

**Who this is for:** a freshly imaged Jetson (or any Jetson not yet on the
cloud dashboard). Three commands and it appears on
[https://34-76-37-31.sslip.io/](https://34-76-37-31.sslip.io/).

For the first-ever cloud setup (creating the VM, DNS, etc.) see
[README.md](README.md). That's a one-time thing; everything below is for the
day-to-day of plugging a new Jetson into the existing relay.

---

## Checklist

### 1. Get the Interceptor repo on the Jetson

```bash
git clone https://github.com/ikh-innovation/Interceptor.git ~/Documents/Interceptor
# or, if already cloned:
cd ~/Documents/Interceptor && git pull
```

### 2. Make sure the Interceptor service is set up

The `install-frpc.sh` installer gates the cloud tunnel on the Interceptor UI
being up on `127.0.0.1:8080`. On an existing Jetson that already runs
`wasp-tracker.service` at boot, skip this step. On a freshly imaged Jetson:

```bash
# One-time: install the wasp-tracker unit (reads src/wasp-tracker.service as a template)
sudo cp ~/Documents/Interceptor/src/wasp-tracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wasp-tracker
```

### 3. Register with the cloud relay

Pick a unique name for this drone. Lowercase letters, digits, hyphens. It
becomes the subdomain, e.g. `wasp-02` → `https://wasp-02.34-76-37-31.sslip.io/`.

```bash
cd ~/Documents/Interceptor
sudo deploy/gcp/jetson/install-frpc.sh --name wasp-02
```

That's it. Expected output:

```
[install] fetching token from https://34-76-37-31.sslip.io/_token
[install] fetching frp v0.61.1 (linux_arm64)
[ok] Jetson 'wasp-02' registered and the Interceptor UI is already up.
     Dashboard: https://34-76-37-31.sslip.io/
     Direct:    https://wasp-02.34-76-37-31.sslip.io/
     Logs:      journalctl -u frpc -f
```

If the installer says `frpc is waiting for the Interceptor UI on 127.0.0.1:8080`,
that's fine — the Jetson will auto-register the moment wasp-tracker finishes
booting (within a few seconds of `MJPEGServer` binding port 8080).

### 4. Verify

- Reboot the Jetson. `frpc` + `wasp-tracker` both auto-start via systemd.
- On any phone/laptop with internet, open `https://34-76-37-31.sslip.io/`.
- Your new drone card appears within seconds of the Jetson being ready. Click → fly.

---

## What the installer does, in case you need to debug

1. Reads [`relay.conf`](jetson/relay.conf) (committed to git) for the relay's
   server address and username.
2. Fetches the frp shared token from `https://<relay>/_token` (no auth; the
   relay dashboard is currently open).
3. Downloads the `frpc` binary for the Jetson's architecture (arm64) from
   GitHub releases.
4. Writes `/etc/frp/frpc.toml` with the drone's name as the subdomain.
5. Writes `/etc/systemd/system/frpc.service` (depends on `wasp-tracker`, waits
   for `127.0.0.1:8080` to accept connections before registering the tunnel).
6. `systemctl enable --now frpc`.

All of it is idempotent — re-run on every `git pull` without consequence.

---

## Common follow-ups

- **Reboot → does it come back?** Yes. Both `wasp-tracker` and `frpc` are
  enabled systemd units. On boot: Jetson gets network → wasp-tracker starts
  Interceptor → `frpc`'s pre-check sees port 8080 open → tunnel opens → drone
  appears on dashboard. No login, no manual step.

- **Rename this Jetson?** Re-run with a different `--name`. The old subdomain
  simply stops appearing on the dashboard once nothing's registered there.

- **Point it at a different relay?** Edit `deploy/gcp/jetson/relay.conf` and
  re-run `install-frpc.sh --name ...`, or pass `--server ...` on the command
  line.

- **Remove it from the fleet?** `sudo systemctl disable --now frpc`. It
  disappears from the dashboard on the next 5-second refresh.

- **See what the tunnel is doing?** `journalctl -u frpc -f`.
