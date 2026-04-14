#!/usr/bin/env bash
# Daily VPS-side backup: SQLite DB + raw TLE archive → local + Cloudflare R2
#
# Requires:
#   - /root/.argus-r2.env exporting AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
#   - aws-cli at /usr/local/bin/aws
#   - argus installed at /opt/argus
#
# Crontab:
#   0 4 * * * /opt/argus/scripts/backup-vps.sh >> /opt/argus/backups/backup.log 2>&1

set -euo pipefail

source /root/.argus-r2.env

APP_DIR="/opt/argus"
DB="$APP_DIR/data/starlink.db"
LOCAL_DIR="$APP_DIR/backups"
RAW_SRC="$APP_DIR/data/raw"

DATE=$(date -u +%Y%m%d)
YESTERDAY=$(date -u --date=yesterday +%Y%m%d)

R2_ENDPOINT="https://f75b3d514123f61fd7ca4a78e029b01a.r2.cloudflarestorage.com"
R2_BUCKET="argus-backups"
AWS="/usr/local/bin/aws"

mkdir -p "$LOCAL_DIR"

# ── SQLite DB backup ──
DB_TMP="/tmp/starlink-${DATE}.db.gz"
sqlite3 "$DB" ".backup /tmp/starlink-backup.db"
gzip -c /tmp/starlink-backup.db > "$DB_TMP"
rm -f /tmp/starlink-backup.db

DB_SIZE=$(ls -lh "$DB_TMP" | awk '{print $5}')
cp "$DB_TMP" "$LOCAL_DIR/starlink-${DATE}.db.gz"
"$AWS" s3 cp "$DB_TMP" "s3://${R2_BUCKET}/starlink-${DATE}.db.gz" \
  --endpoint-url "$R2_ENDPOINT" --region auto
rm -f "$DB_TMP"
echo "[BACKUP] $DATE db: $DB_SIZE → local + R2"

# ── Raw Celestrak archive backup ──
# The fetcher writes raw TLEs to data/raw/YYYYMMDD/*.tle.gz. These are the
# "never lose an update" insurance — if parsing has bugs or Celestrak
# changes format, history can be replayed from these files. Each day's
# directory is immutable after the day ends, so we upload yesterday's
# tarball exactly once.
RAW_YESTERDAY="$RAW_SRC/$YESTERDAY"
if [ -d "$RAW_YESTERDAY" ]; then
    RAW_TAR="/tmp/raw-${YESTERDAY}.tar.gz"
    tar -czf "$RAW_TAR" -C "$RAW_SRC" "$YESTERDAY"
    RAW_SIZE=$(ls -lh "$RAW_TAR" | awk '{print $5}')
    "$AWS" s3 cp "$RAW_TAR" "s3://${R2_BUCKET}/raw-${YESTERDAY}.tar.gz" \
      --endpoint-url "$R2_ENDPOINT" --region auto
    rm -f "$RAW_TAR"
    echo "[BACKUP] $YESTERDAY raw: $RAW_SIZE → R2"
else
    echo "[BACKUP] $YESTERDAY raw: (no archive dir, skipping)"
fi
