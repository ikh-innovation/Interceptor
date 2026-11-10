#!/usr/bin/env bash
# Interceptor — register a Jetson with the GCP relay.
#
# Simplest form (reads relay.conf, prompts for dashboard password):
#   sudo ./install-frpc.sh --name wasp-03
#
# Full form (bypass prompts / relay.conf):
#   sudo ./install-frpc.sh --name wasp-03 \
#     --server 34-76-37-31.sslip.io --user pilot --password 1234iknowhow
#   sudo ./install-frpc.sh --name wasp-03 --token <raw-frp-token>
#
# Idempotent: safe to re-run after git pull or on a freshly imaged Jetson.
set -euo pipefail

NAME=""
SERVER=""
USER_NAME=""
PASSWORD=""
TOKEN=""
LOCAL_PORT="8080"
FRP_VERSION="0.61.1"

die() { echo "error: $*" >&2; exit 1; }
usage() { sed -n '2,13p' "$0"; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --name)         NAME="$2"; shift 2 ;;
    --server)       SERVER="$2"; shift 2 ;;
    --user)         USER_NAME="$2"; shift 2 ;;
    --password)     PASSWORD="$2"; shift 2 ;;
    --token)        TOKEN="$2"; shift 2 ;;
    --local-port)   LOCAL_PORT="$2"; shift 2 ;;
    --frp-version)  FRP_VERSION="$2"; shift 2 ;;
    -h|--help)      usage ;;
    *) die "unknown arg: $1" ;;
  esac
done

[ -n "$NAME" ]                || die "--name is required (e.g. wasp-03)"
[ "$(id -u)" = "0" ]          || die "run with sudo"
echo "$NAME" | grep -Eq '^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$' \
  || die "name must be a DNS label: lowercase alnum + hyphens, 1-63 chars"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$SCRIPT_DIR/relay.conf"
TEMPLATE="$SCRIPT_DIR/frpc.toml.template"
UNIT_TEMPLATE="$SCRIPT_DIR/frpc.service.template"
[ -f "$TEMPLATE" ]      || die "missing $TEMPLATE"
[ -f "$UNIT_TEMPLATE" ] || die "missing $UNIT_TEMPLATE"

# --- Load relay.conf for defaults -------------------------------------------
if [ -f "$CONF" ]; then
  # shellcheck source=/dev/null
  . "$CONF"
  [ -z "$SERVER" ]    && SERVER="${RELAY_SERVER:-}"
  [ -z "$USER_NAME" ] && USER_NAME="${RELAY_USER:-pilot}"
fi
[ -z "$USER_NAME" ] && USER_NAME="pilot"

[ -n "$SERVER" ] || die "no --server given and $CONF has no RELAY_SERVER"

# --- Fetch token from the relay (no auth; dashboard is open) ---------------
if [ -z "$TOKEN" ]; then
  if [ -n "$PASSWORD" ]; then
    # Legacy path: dashboard is open now, but keep --password working
    # in case someone re-enables basic auth on the relay later.
    TOKEN=$(curl -fsS -u "$USER_NAME:$PASSWORD" "https://$SERVER/_token") \
      || die "failed to fetch token (wrong password? relay unreachable?)"
  else
    echo "[install] fetching token from https://$SERVER/_token"
    TOKEN=$(curl -fsS "https://$SERVER/_token") \
      || die "failed to fetch token (relay unreachable?)"
  fi
  [ -n "$TOKEN" ] || die "relay returned empty token"
fi

# --- Architecture -----------------------------------------------------------
case "$(uname -m)" in
  aarch64|arm64) FRP_ARCH="linux_arm64" ;;
  x86_64)        FRP_ARCH="linux_amd64" ;;
  armv7l)        FRP_ARCH="linux_arm" ;;
  *) die "unsupported arch: $(uname -m)" ;;
esac

# --- Install frpc binary ----------------------------------------------------
if [ ! -x /usr/local/bin/frpc ] || ! /usr/local/bin/frpc --version 2>/dev/null | grep -q "$FRP_VERSION"; then
  echo "[install] fetching frp v$FRP_VERSION ($FRP_ARCH)"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  cd "$TMP"
  URL="https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/frp_${FRP_VERSION}_${FRP_ARCH}.tar.gz"
  curl -fsSL "$URL" -o frp.tgz
  tar -xzf frp.tgz
  install -m 0755 "frp_${FRP_VERSION}_${FRP_ARCH}/frpc" /usr/local/bin/frpc
  cd /
fi

# --- Config -----------------------------------------------------------------
install -d -m 0755 /etc/frp
TMP_CONF="$(mktemp)"
sed \
  -e "s|__SERVER__|$SERVER|g" \
  -e "s|__TOKEN__|$TOKEN|g" \
  -e "s|__NAME__|$NAME|g" \
  -e "s|__LOCAL_PORT__|$LOCAL_PORT|g" \
  "$TEMPLATE" > "$TMP_CONF"
install -m 0640 "$TMP_CONF" /etc/frp/frpc.toml
rm -f "$TMP_CONF"

# --- Systemd unit -----------------------------------------------------------
TMP_UNIT="$(mktemp)"
sed -e "s|__LOCAL_PORT__|$LOCAL_PORT|g" "$UNIT_TEMPLATE" > "$TMP_UNIT"
install -m 0644 "$TMP_UNIT" /etc/systemd/system/frpc.service
rm -f "$TMP_UNIT"
systemctl daemon-reload
systemctl enable frpc.service
systemctl restart frpc.service

# --- Status -----------------------------------------------------------------
sleep 1
state="$(systemctl is-active frpc.service || true)"
echo ""
if [ "$state" = "active" ]; then
  echo "[ok] Jetson '$NAME' registered and the Interceptor UI is already up."
elif [ "$state" = "activating" ]; then
  echo "[ok] Jetson '$NAME' registered."
  echo "     frpc is waiting for the Interceptor UI on 127.0.0.1:$LOCAL_PORT."
  echo "     It will go live on the dashboard the moment the service binds that port."
else
  echo "[warn] frpc is '$state'. Check: journalctl -u frpc -n 50"
  exit 1
fi
echo "     Dashboard: https://$SERVER/"
echo "     Direct:    https://$NAME.$SERVER/"
echo "     Logs:      journalctl -u frpc -f"
