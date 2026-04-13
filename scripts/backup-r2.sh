#!/usr/bin/env bash
# Daily backup to Cloudflare R2 (S3-compatible)
# Requires: ~/.argus-r2.env with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
# Add to crontab: 0 4 * * * /Users/yong/projects/selene-insight/scripts/backup-r2.sh

set -euo pipefail

# Load R2 credentials from env file
source "$HOME/.argus-r2.env"

REMOTE="root@76.13.118.240"
REMOTE_DB="/opt/selene-insight/data/starlink.db"
DATE=$(date +%Y%m%d)
TMPFILE="/tmp/starlink-$DATE.db.gz"
R2_ENDPOINT="https://f75b3d514123f61fd7ca4a78e029b01a.r2.cloudflarestorage.com"
R2_BUCKET="argus-backups"

# Safe copy + compress
ssh "$REMOTE" "sqlite3 $REMOTE_DB '.backup /tmp/starlink-backup.db' && gzip -c /tmp/starlink-backup.db" > "$TMPFILE"
ssh "$REMOTE" "rm -f /tmp/starlink-backup.db"

SIZE=$(ls -lh "$TMPFILE" | awk '{print $5}')

# Upload to R2
aws s3 cp "$TMPFILE" "s3://${R2_BUCKET}/starlink-${DATE}.db.gz" \
  --endpoint-url "$R2_ENDPOINT" \
  --region auto

rm -f "$TMPFILE"

echo "[R2 BACKUP] $DATE: $SIZE → s3://argus-backups/starlink-${DATE}.db.gz"
