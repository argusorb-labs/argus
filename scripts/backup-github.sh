#!/usr/bin/env bash
# Weekly backup: upload SQLite DB as GitHub Release asset
# Add to crontab: 0 4 * * 0 /Users/yong/projects/selene-insight/scripts/backup-github.sh

set -euo pipefail

REMOTE="root@76.13.118.240"
REMOTE_DB="/opt/selene-insight/data/starlink.db"
REPO="Aether-Labs/selene-insight"
DATE=$(date +%Y%m%d)
TAG="backup-$DATE"
TMPFILE="/tmp/starlink-$DATE.db.gz"

# Safe copy + compress
ssh "$REMOTE" "sqlite3 $REMOTE_DB '.backup /tmp/starlink-backup.db' && gzip -c /tmp/starlink-backup.db" > "$TMPFILE"
ssh "$REMOTE" "rm -f /tmp/starlink-backup.db"

SIZE=$(ls -lh "$TMPFILE" | awk '{print $5}')
echo "[BACKUP] Compressed: $SIZE"

# Create GitHub release with the DB as asset
gh release create "$TAG" "$TMPFILE" \
  --repo "$REPO" \
  --title "Data Backup $DATE" \
  --notes "Starlink TLE database backup. $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --latest=false

rm -f "$TMPFILE"

# Keep only last 8 weekly backups on GitHub
RELEASES=$(gh release list --repo "$REPO" --limit 100 --json tagName -q '.[].tagName' | grep "^backup-" | sort -r)
COUNT=0
for rel in $RELEASES; do
  COUNT=$((COUNT + 1))
  if [ $COUNT -gt 8 ]; then
    gh release delete "$rel" --repo "$REPO" --yes --cleanup-tag 2>/dev/null || true
  fi
done

echo "[BACKUP] GitHub release $TAG created"
