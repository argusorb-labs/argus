#!/usr/bin/env bash
# Oracle Cloud Free Tier (ARM) — one-shot deploy script
# Run as root on a fresh Ubuntu 22.04+ aarch64 instance
#
# Usage: ssh ubuntu@<IP> 'bash -s' < deploy/setup.sh

set -euo pipefail

REPO="git@github.com:aether-labs/selene-insight.git"
APP_DIR="/opt/selene-insight"

echo "=== [1/5] Install Docker ==="
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
    usermod -aG docker ubuntu
fi

echo "=== [2/5] Open firewall ports (8000=API, 8080=Web) ==="
iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
# Persist iptables rules across reboot
if command -v netfilter-persistent &>/dev/null; then
    netfilter-persistent save
else
    apt-get install -y iptables-persistent
    netfilter-persistent save
fi

echo "=== [3/5] Clone repo ==="
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git pull
else
    git clone "$REPO" "$APP_DIR"
    cd "$APP_DIR"
fi

echo "=== [4/5] Build and start ==="
docker compose up -d --build

echo "=== [5/5] Verify ==="
sleep 5
docker compose ps
curl -sf http://localhost:8000/api/status && echo " <- API OK"

echo ""
echo "Done. Open these ports in Oracle VCN Security List:"
echo "  - 8000/tcp (API)"
echo "  - 8080/tcp (Web dashboard)"
echo ""
echo "Dashboard: http://<PUBLIC_IP>:8080"
echo "API:       http://<PUBLIC_IP>:8000/api/status"
