# Add a new Jetson to the fleet

**Who this is for:** a freshly imaged Jetson (or any Jetson not yet on the
cloud dashboard). Follow the checklist and it appears on
[https://34-76-37-31.sslip.io/](https://34-76-37-31.sslip.io/) with a live
heartbeat, uploaded flight logs, and a Vertex AI Greek report per flight.

For the first-ever cloud setup (creating the VM, DNS, etc.) see
[README.md](README.md). That's a one-time thing; everything below is for the
day-to-day of plugging a new Jetson into the existing relay.

---

## The one rule: pick a unique name and use it everywhere

Every Jetson in the fleet needs a **unique name** — lowercase letters, digits,
hyphens, 1–63 chars (e.g. `jetson`, `wasp-01`, `wasp-02`, `prototype-a`). That
same name goes into **two places**:

1. `sudo install-frpc.sh --name <NAME>` — becomes the cockpit subdomain:
   `https://<NAME>.34-76-37-31.sslip.io/`
2. `JETSON_ID=<NAME>` in the wasp-tracker runtime config — the key for the
   Firestore drone document and its flights.

If these two names disagree, the dashboard lists the drone but the "Cockpit"
button opens the wrong subdomain (404). **Keep them identical.**

---

## Checklist

### 1. Get the Interceptor repo on the Jetson

```bash
git clone https://github.com/ikh-innovation/Interceptor.git ~/Documents/Interceptor
# or, if already cloned:
cd ~/Documents/Interceptor && git pull
```

### 2. Install the wasp-tracker service (fresh Jetson only)

If this Jetson already runs `wasp-tracker` at boot, skip to step 3.

```bash
sudo cp ~/Documents/Interceptor/src/wasp-tracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wasp-tracker
```

### 3. Install the Firestore client in the yolo venv

The tracker pushes logs to GCP through this. Without it, uploads are silently
disabled and the Jetson never appears on the dashboard.

```bash
~/Documents/Interceptor/.env/yolo/bin/pip install google-cloud-firestore
```

### 4. Give the Jetson GCP credentials

The tracker runs as your user, so Application Default Credentials (ADC) is
the simplest path:

```bash
gcloud auth application-default login
```

Follow the browser prompt. The credentials land in
`~/.config/gcloud/application_default_credentials.json`, which the Firestore
client picks up automatically.

> **Headless Jetson?** Use a service account key instead:
> ```bash
> # On your workstation, once per Jetson:
> gcloud iam service-accounts keys create ~/jetson-sa.json \
>   --iam-account=489394273704-compute@developer.gserviceaccount.com
> # Copy jetson-sa.json to the Jetson, then:
> echo 'Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/iknowhow/jetson-sa.json"' \
>   | sudo tee -a /home/iknowhow/.config/systemd/user/wasp-tracker.service.d/runtime.conf
> ```

### 5. Point the tracker at GCP and name the Jetson

Create or edit the runtime drop-in:

```bash
mkdir -p ~/.config/systemd/user/wasp-tracker.service.d
cat > ~/.config/systemd/user/wasp-tracker.service.d/runtime.conf <<'EOF'
[Service]
Environment="WASP_YOLO_PY=/home/iknowhow/Documents/Interceptor/.env/yolo/bin/python"
Environment="WASP_APP=mission"
Environment="CONTROL_LAW=intercept"
Environment="CONTROL_AXES=roll,pitch,yaw,throttle"
Environment="FC_STEER_MODES=ALT_HOLD,ACRO"
Environment="TRACKER_HEADLESS=1"
Environment="WASP_MODEL=/home/iknowhow/Documents/Interceptor/runs/detect/v0/weights/best.engine"
Environment="CV_THREADS=4"
Environment="STREAM_FPS=15"
Environment="WASP_RECORD_DIR=/home/iknowhow/Documents/Interceptor/src/data/video"
Environment="WASP_RECORD_CONTINUOUS=1"
Environment="GCP_PROJECT=ikh-drone-cloud"
Environment="JETSON_ID=<NAME>"
EOF
systemctl --user daemon-reload
systemctl --user restart wasp-tracker
```

Replace `<NAME>` with the unique name you chose. **Remember this — you need
the same string in step 6.**

### 6. Register with the cloud relay (frpc tunnel for the cockpit)

```bash
cd ~/Documents/Interceptor
sudo deploy/gcp/jetson/install-frpc.sh --name <NAME>
```

Use **the same `<NAME>`** as in step 5. That makes the dashboard's "Cockpit"
button work, because the dashboard opens
`https://<JETSON_ID>.34-76-37-31.sslip.io/` by convention.

Expected output:

```
[install] fetching token from https://34-76-37-31.sslip.io/_token
[install] fetching frp v0.61.1 (linux_arm64)
[ok] Jetson '<NAME>' registered and the Interceptor UI is already up.
     Dashboard: https://34-76-37-31.sslip.io/
     Direct:    https://<NAME>.34-76-37-31.sslip.io/
     Logs:      journalctl -u frpc -f
```

### 7. Verify

Within ~30 seconds:

- `https://34-76-37-31.sslip.io/` → your drone card with a green dot, last
  seen "just now".
- Click **Cockpit** → the Interceptor UI opens on
  `<NAME>.34-76-37-31.sslip.io`.
- Fly something. On DISARM, within ~1 minute, the flight appears with a
  **REPORT** badge and a Greek Vertex AI summary.

Flight logs live both locally (`~/Documents/Interceptor/src/data/flights/*/`)
and in Firestore (`flight_logs/<NAME>/flights/<flight_id>`). Videos (with
bboxes/HUD burned in) stay local at
`~/Documents/Interceptor/src/data/video/rec_*.mp4`.

---

## Renaming a Jetson

Change in both places, in order:

```bash
# 1. Stop tracker so the stale "online" heartbeat doesn't keep flipping
systemctl --user stop wasp-tracker

# 2. Change JETSON_ID in the drop-in
sed -i 's/JETSON_ID=.*/JETSON_ID=<new-name>"/' \
  ~/.config/systemd/user/wasp-tracker.service.d/runtime.conf

# 3. Re-register the tunnel under the new subdomain
sudo ~/Documents/Interceptor/deploy/gcp/jetson/install-frpc.sh --name <new-name>

# 4. Restart tracker
systemctl --user daemon-reload
systemctl --user restart wasp-tracker

# 5. Optional: delete the stale old drone document from Firestore
#    (the old flights stay under <old-name> forever otherwise)
```

---

## Debugging

**"I don't see my Jetson on the dashboard."**

```bash
# Is the tracker running?
systemctl --user status wasp-tracker

# Does it see the GCP env?
systemctl --user show wasp-tracker -p Environment | grep -o 'JETSON_ID=[^"]*'

# Did the uploader connect?
tail -200 ~/wasp-logs/wasp-tracker.log | grep -iE 'uploader|firestore|heartbeat'
# Expect: "LogUploader: connected to Firestore project=ikh-drone-cloud"
# A "ModuleNotFoundError: google.cloud" means step 3 was skipped.
# A "DefaultCredentialsError" means step 4 was skipped.
```

**"Dashboard shows the drone but the Cockpit button 404s."**

`JETSON_ID` and the frpc `--name` disagree. Re-run step 6 with the same name
as `JETSON_ID`.

**"The drone shows OFFLINE but I know it's running."**

The heartbeat writes every 30 s. If more than ~2 minutes have passed since
`last_seen`, the uploader is probably broken — check the log line above.

**"Flight completed but no REPORT appeared."**

The Greek report is a Cloud Function triggered by Firestore. Check it:
```bash
gcloud functions logs read flight-report-generator \
  --project=ikh-drone-cloud --region=europe-west1 --limit=20
```
A failed report does not block the flight record — you can also click
"Generate Report" in the flight detail modal to retry on demand.
