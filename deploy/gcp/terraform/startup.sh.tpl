#!/usr/bin/env bash
# VM bootstrap: installs frp server, Caddy with HTTPS + basic auth, and the
# "Drones Online" dashboard. Idempotent — Terraform re-runs it on metadata change.
set -euo pipefail

exec > >(tee -a /var/log/drones-startup.log) 2>&1
echo "[startup] $(date -Is) starting"

export DEBIAN_FRONTEND=noninteractive

# --- Packages ---------------------------------------------------------------
apt-get update
apt-get install -y curl ca-certificates debian-keyring debian-archive-keyring apt-transport-https gnupg jq

# Caddy official repo
if [ ! -f /usr/share/keyrings/caddy-stable-archive-keyring.gpg ]; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | tee /etc/apt/sources.list.d/caddy-stable.list
fi
apt-get update
apt-get install -y caddy

# --- frp server -------------------------------------------------------------
FRP_VERSION="0.61.1"
FRP_ARCH="linux_amd64"
if [ ! -x /usr/local/bin/frps ] || ! /usr/local/bin/frps --version 2>/dev/null | grep -q "$FRP_VERSION"; then
  cd /tmp
  rm -rf "frp_$${FRP_VERSION}_$${FRP_ARCH}" "frp_$${FRP_VERSION}_$${FRP_ARCH}.tar.gz"
  curl -fsSL "https://github.com/fatedier/frp/releases/download/v$${FRP_VERSION}/frp_$${FRP_VERSION}_$${FRP_ARCH}.tar.gz" -o frp.tar.gz
  tar -xzf frp.tar.gz
  install -m 0755 "frp_$${FRP_VERSION}_$${FRP_ARCH}/frps" /usr/local/bin/frps
fi

# --- frps config ------------------------------------------------------------
install -d -m 0755 /etc/frp
cat > /etc/frp/frps.toml <<FRPS_EOF
bindAddr    = "0.0.0.0"
bindPort    = 7000

# HTTP vhost — Caddy terminates TLS and reverse-proxies into this port,
# frp routes to the right Jetson tunnel based on Host header.
vhostHTTPPort = 7080
subDomainHost = "${domain}"

# Admin API — Caddy proxies /api/* here so the dashboard can list tunnels.
webServer.addr     = "127.0.0.1"
webServer.port     = 7500
webServer.user     = "${frp_admin_user}"
webServer.password = "${frp_admin_password}"

auth.method = "token"
auth.token  = "${frp_token}"

log.to      = "/var/log/frps.log"
log.level   = "info"
log.maxDays = 7
FRPS_EOF
chmod 0640 /etc/frp/frps.toml

cat > /etc/systemd/system/frps.service <<'FRPS_UNIT'
[Unit]
Description=frp server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/frps -c /etc/frp/frps.toml
Restart=on-failure
RestartSec=3
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
FRPS_UNIT

# --- Dashboard --------------------------------------------------------------
install -d -m 0755 /var/www/drones
echo '${dashboard_html_b64}' | base64 -d > /var/www/drones/index.html
chmod 0644 /var/www/drones/index.html

# --- frp token served over HTTPS (behind basic auth) -----------------------
# The Jetson installer fetches this with the dashboard password, so operators
# never have to copy the token by hand.
printf '%s' '${frp_token}' > /var/www/drones/_token
chmod 0644 /var/www/drones/_token

# --- TLS "ask" service ------------------------------------------------------
# Caddy on-demand TLS calls this before fetching a cert; we only say yes for
# names under our own domain, so random probes can't exhaust Let's Encrypt limits.
cat > /usr/local/bin/tls-ask.py <<ASK_EOF
#!/usr/bin/env python3
import http.server, socketserver, urllib.parse, re
DOMAIN = "${domain}"
PAT = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\." + re.escape(DOMAIN) + r"\$")
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        d = (q.get("domain") or [""])[0].lower()
        ok = d == DOMAIN or bool(PAT.match(d))
        self.send_response(200 if ok else 404)
        self.end_headers()
    def log_message(self, *a, **k): pass
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", 9999), H) as s:
    s.serve_forever()
ASK_EOF
chmod 0755 /usr/local/bin/tls-ask.py

cat > /etc/systemd/system/tls-ask.service <<'ASK_UNIT'
[Unit]
Description=TLS on-demand ask endpoint for Caddy
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/tls-ask.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
ASK_UNIT

systemctl daemon-reload
systemctl enable --now tls-ask.service
systemctl restart tls-ask.service

# --- Caddy config -----------------------------------------------------------
cat > /etc/caddy/Caddyfile <<CADDY_EOF
{
    email ${letsencrypt_email}
    on_demand_tls {
        ask http://127.0.0.1:9999/check
    }
}

# Dashboard host — lists online Jetsons. OPEN: anyone with the URL can access.
${domain} {
    # Proxy frp admin API so the dashboard can fetch the tunnel list.
    # Caddy injects the admin creds; the browser never sees them.
    handle /api/* {
        reverse_proxy 127.0.0.1:7500 {
            header_up Authorization "Basic ${frp_admin_basic_b64}"
        }
    }

    root * /var/www/drones
    file_server
}

# Per-Jetson subdomain — Caddy terminates TLS, frp routes to the right tunnel.
*.${domain} {
    tls {
        on_demand
    }
    reverse_proxy 127.0.0.1:7080 {
        header_up Host {host}
        header_up X-Forwarded-Host {host}
        # MJPEG / SSE: flush every write instead of buffering, or the video
        # looks frozen in the browser. Long timeouts for the same reason.
        flush_interval -1
        transport http {
            read_timeout 24h
            write_timeout 24h
            keepalive 2m
        }
    }
}
CADDY_EOF

systemctl daemon-reload
systemctl enable --now frps.service
systemctl restart frps.service

systemctl enable --now caddy
systemctl restart caddy

echo "[startup] $(date -Is) done"
