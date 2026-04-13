#!/usr/bin/env bash
# Daily local backup of Starlink SQLite database
# Add to crontab: 0 3 * * * /Users/yong/projects/selene-insight/scripts/backup-local.sh

set -euo pipefail

REMOTE="root@76.13.118.240"
REMOTE_DB="/opt/selene-insight/data/starlink.db"
LOCAL_DIR="/Users/yong/projects/selene-insight/backups"
DATE=$(date +%Y%m%d)

# SQLite safe copy: use .backup command on remote to avoid copying mid-write
ssh "$REMOTE" "sqlite3 $REMOTE_DB '.backup /tmp/starlink-backup.db'" 2>/dev/null

scp -q "$REMOTE:/tmp/starlink-backup.db" "$LOCAL_DIR/starlink-$DATE.db"
ssh "$REMOTE" "rm -f /tmp/starlink-backup.db"

# Keep all backups permanently (data is the moat)

SIZE=$(ls -lh "$LOCAL_DIR/starlink-$DATE.db" | awk '{print $5}')
echo "[BACKUP] $DATE: $SIZE → $LOCAL_DIR/starlink-$DATE.db"
