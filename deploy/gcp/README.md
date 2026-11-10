# Interceptor cloud relay (GCP, IP-only, no domain)

Deploy a GCP relay so any Jetson with internet appears on a web dashboard, and a
pilot on any phone/laptop with internet can open the drone's cockpit over HTTPS
— all without buying a domain. Uses [sslip.io](https://sslip.io): the relay's
public IP `1.2.3.4` is automatically reachable as `1-2-3-4.sslip.io` with full
Let's Encrypt HTTPS.

**No product code is changed** — the Jetson's `MJPEGServer` keeps serving
`127.0.0.1:8080`; a tiny `frpc` sidecar tunnels that out to the cloud.

```
  Phone / laptop  ──HTTPS──▶  https://1-2-3-4.sslip.io/  (GCP VM)
                                 │
                                 │ Caddy (TLS + basic auth)
                                 │   ├── /         → "Drones Online" dashboard
                                 │   ├── /api/*    → frps admin API
                                 │   └── *.<ip>.sslip.io → frps HTTP vhost
                                 │                             │
                                 ▼                             │ reverse tunnel
                               frps ─── routes by Host ──────▶ frpc → 127.0.0.1:8080
                                                                     (Interceptor UI)
```

---

## Terminal-only walkthrough

Do these on your laptop — the current working directory is
`/home/iknowhow/Documents/Interceptor`.

### 0. Safety first

You pasted a Google password into chat. **Change it now**:

```bash
xdg-open https://myaccount.google.com/security
```

gcloud never uses your password — it opens a browser-based OAuth flow once.

### 1. Install tools

```bash
sudo apt update
sudo apt install -y terraform python3-bcrypt jq
curl -sSL https://sdk.cloud.google.com | bash
exec -l $SHELL        # reload PATH so gcloud is found
```

Verify:

```bash
terraform -version
gcloud --version
```

### 2. Log in to GCP

```bash
gcloud auth login                              # opens a browser once
gcloud auth application-default login          # same, for Terraform
gcloud config set project ikh-drone-cloud
```

### 3. Attach billing (one unavoidable browser step)

If you've never added a card to GCP, you need to create a billing account
*once* in the console:

```bash
xdg-open https://console.cloud.google.com/billing/create
```

After that, everything is CLI:

```bash
gcloud billing accounts list
# copy the ACCOUNT_ID that shows up, e.g. 0X0X0X-0X0X0X-0X0X0X
gcloud billing projects link ikh-drone-cloud --billing-account=<ACCOUNT_ID>
```

Enable the APIs Terraform will touch (Terraform does this too, but doing it now
surfaces permission issues early):

```bash
gcloud services enable compute.googleapis.com
```

### 4. Generate your secrets

```bash
cd deploy/gcp/terraform

# Dashboard password (pick your own, keep it)
DASH_PASSWORD="change-me-to-something-strong"
BCRYPT_HASH=$(python3 -c "import bcrypt,sys; print(bcrypt.hashpw(sys.argv[1].encode(), bcrypt.gensalt()).decode())" "$DASH_PASSWORD")
FRP_TOKEN=$(openssl rand -hex 32)
FRP_ADMIN_PASSWORD=$(openssl rand -hex 16)

echo "---- SAVE THESE ----"
echo "Dashboard login:  pilot / $DASH_PASSWORD"
echo "frp token:        $FRP_TOKEN"
echo "frp admin pass:   $FRP_ADMIN_PASSWORD"
echo "bcrypt hash:      $BCRYPT_HASH"
echo "--------------------"
```

Create `terraform.tfvars` from the template:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Open it with your editor and paste in the values printed above
(`basic_auth_bcrypt`, `frp_token`, `frp_admin_password`). Set
`letsencrypt_email` to an email you actually control (e.g.
`achitos@ikhdefencelabs.com`).

### 5. Launch

```bash
terraform init
terraform apply        # type "yes"
```

Takes ~3 min. When it finishes:

```bash
terraform output dashboard_url
```

Example: `https://34-76-12-55.sslip.io/`

Open it on your phone or laptop. First load takes ~20 s while Caddy fetches
the Let's Encrypt certificate. You'll get a basic-auth prompt — user `pilot`,
password = the `DASH_PASSWORD` you picked above. The dashboard shows "No drones
online yet" until you register one.

### 6. Register a Jetson

`terraform apply` writes `deploy/gcp/jetson/relay.conf` (committed to git) with
the relay's address. Any Jetson that did `git pull` already has it.

On each Jetson, one command:

```bash
cd ~/Documents/Interceptor && git pull
sudo deploy/gcp/jetson/install-frpc.sh --name wasp-01
# Prompt: Dashboard password  →  type the password you set in step 4
```

The installer fetches the frp token from the cloud over HTTPS (using that
password), writes it into the Jetson's `frpc` config, and starts the service.
Within seconds, the Jetson appears on the dashboard. Click → fly.

---

## Day-to-day

- **Open the cockpit on your phone**: dashboard URL → login → pick a drone.
- **Add a new Jetson**: clone repo, run `install-frpc.sh`. Nothing else.
- **Rotate the dashboard password**: regenerate `BCRYPT_HASH` (step 4), update
  `terraform.tfvars`, `terraform apply`. Caddy reloads, no downtime.
- **Retire a Jetson**: on the Jetson, `sudo systemctl disable --now frpc`.
  It disappears from the dashboard on next refresh.
- **Teardown everything**: `terraform destroy`. Removes the VM, IP, firewall.
- **Buy a real domain later**: swap `local.domain` in `main.tf` back for a
  `var.domain`, add Cloud DNS. The whole rest of the stack is unchanged.

---

## Troubleshooting

```bash
# Can't open dashboard URL
gcloud compute ssh drones-relay --zone europe-west1-b
sudo tail -f /var/log/drones-startup.log
sudo journalctl -u caddy -n 50
sudo journalctl -u frps -n 50
```

```bash
# Jetson not appearing on dashboard
journalctl -u frpc -n 50       # should see "start proxy success"
# If it says "login to server failed", the frp token is wrong.
# If it says "dial tcp ... i/o timeout", the Jetson can't reach the VM on :7000.
```

---

## Security notes

- **TLS**: Let's Encrypt via Caddy on-demand; a local "ask" service limits cert
  requests to names under your own IP's sslip.io hostname.
- **Auth**: single shared username/password at Caddy. Rotate when leaked.
- **Tunnel auth**: `frp_token` is a shared secret. Kept in `terraform.tfvars`,
  which is gitignored.
- **Public attack surface**: 22 (SSH via OS Login), 80/443 (Caddy), 7000 (frps,
  rejects anyone without the token).

---

## Files

```
deploy/gcp/
├── README.md                      — this runbook
├── terraform/
│   ├── versions.tf · variables.tf · main.tf · outputs.tf
│   ├── startup.sh.tpl             — VM bootstrap (frps + Caddy + dashboard)
│   └── terraform.tfvars.example   — copy to terraform.tfvars
├── vm/
│   └── dashboard.html             — "Drones Online" page baked into the VM
└── jetson/
    ├── install-frpc.sh            — one-shot idempotent registrar
    ├── frpc.toml.template         — rendered by the installer
    └── frpc.service.template      — systemd unit; waits for Interceptor on :8080 before tunnelling
```
