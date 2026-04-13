"""SQLite-backed persistent store for Starlink TLE data.

Three tables:
- tle: every TLE update, deduplicated by (norad_id, epoch_jd)
- satellite: metadata, shell classification, status
- anomaly: detected orbital events

This is the data moat — every TLE ever seen is stored permanently.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = os.environ.get("SELENE_DB_PATH", "data/starlink.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tle (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    norad_id INTEGER NOT NULL,
    epoch_jd REAL NOT NULL,
    fetched_at REAL NOT NULL,
    line1 TEXT NOT NULL,
    line2 TEXT NOT NULL,
    inclination REAL,
    mean_motion REAL,
    eccentricity REAL,
    UNIQUE(norad_id, epoch_jd)
);
CREATE INDEX IF NOT EXISTS idx_tle_norad ON tle(norad_id, epoch_jd DESC);
CREATE INDEX IF NOT EXISTS idx_tle_fetched ON tle(fetched_at);

CREATE TABLE IF NOT EXISTS satellite (
    norad_id INTEGER PRIMARY KEY,
    name TEXT,
    intl_designator TEXT,
    shell_km REAL,
    launch_group TEXT,
    first_seen REAL,
    last_seen REAL,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS anomaly (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    norad_id INTEGER NOT NULL,
    detected_at REAL NOT NULL,
    anomaly_type TEXT NOT NULL,
    details TEXT,
    altitude_before_km REAL,
    altitude_after_km REAL,
    FOREIGN KEY(norad_id) REFERENCES satellite(norad_id)
);
CREATE INDEX IF NOT EXISTS idx_anomaly_time ON anomaly(detected_at DESC);
"""


class StarlinkStore:
    """Thread-safe SQLite store for Starlink constellation data."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA)
            conn.close()

    # ── TLE operations ──

    def upsert_tles(self, tles: list[dict]) -> int:
        """Insert TLEs, skipping duplicates. Returns count of new inserts."""
        now = time.time()
        new_count = 0
        with self._lock:
            conn = self._get_conn()
            for t in tles:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO tle
                           (norad_id, epoch_jd, fetched_at, line1, line2,
                            inclination, mean_motion, eccentricity)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            t["norad_id"], t["epoch_jd"], now,
                            t["line1"], t["line2"],
                            t.get("inclination"), t.get("mean_motion"),
                            t.get("eccentricity"),
                        ),
                    )
                    if conn.total_changes:
                        new_count += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()

            # Update satellite metadata
            for t in tles:
                conn.execute(
                    """INSERT INTO satellite (norad_id, name, intl_designator,
                           shell_km, launch_group, first_seen, last_seen, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
                       ON CONFLICT(norad_id) DO UPDATE SET
                           name=excluded.name,
                           shell_km=excluded.shell_km,
                           last_seen=excluded.last_seen""",
                    (
                        t["norad_id"], t.get("name", ""),
                        t.get("intl_designator", ""),
                        t.get("shell_km"), t.get("launch_group"),
                        now, now,
                    ),
                )
            conn.commit()
            conn.close()
        return new_count

    def get_latest_tles(self) -> list[dict]:
        """Get the most recent TLE for each satellite."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT t.norad_id, t.line1, t.line2, t.epoch_jd,
                      t.inclination, t.mean_motion, t.eccentricity,
                      s.name, s.shell_km, s.status
               FROM tle t
               JOIN (SELECT norad_id, MAX(epoch_jd) as max_epoch
                     FROM tle GROUP BY norad_id) latest
               ON t.norad_id = latest.norad_id AND t.epoch_jd = latest.max_epoch
               LEFT JOIN satellite s ON t.norad_id = s.norad_id"""
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_satellite_history(self, norad_id: int, limit: int = 100) -> list[dict]:
        """Get TLE history for a single satellite."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT t.*, s.name, s.shell_km, s.status
               FROM tle t
               LEFT JOIN satellite s ON t.norad_id = s.norad_id
               WHERE t.norad_id = ?
               ORDER BY t.epoch_jd DESC LIMIT ?""",
            (norad_id, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_satellite(self, norad_id: int) -> dict | None:
        """Get satellite metadata."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM satellite WHERE norad_id = ?", (norad_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    # ── Anomaly operations ──

    def insert_anomaly(self, anomaly: dict) -> None:
        """Store a detected anomaly."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO anomaly
                   (norad_id, detected_at, anomaly_type, details,
                    altitude_before_km, altitude_after_km)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    anomaly["norad_id"], anomaly.get("detected_at", time.time()),
                    anomaly["anomaly_type"], anomaly.get("details", ""),
                    anomaly.get("altitude_before_km"),
                    anomaly.get("altitude_after_km"),
                ),
            )
            conn.commit()
            conn.close()

    def get_anomalies(self, limit: int = 50) -> list[dict]:
        """Get recent anomalies."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT a.*, s.name FROM anomaly a
               LEFT JOIN satellite s ON a.norad_id = s.norad_id
               ORDER BY a.detected_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Stats ──

    @property
    def stats(self) -> dict:
        conn = self._get_conn()
        sat_count = conn.execute("SELECT COUNT(*) FROM satellite").fetchone()[0]
        tle_count = conn.execute("SELECT COUNT(*) FROM tle").fetchone()[0]
        anomaly_count = conn.execute("SELECT COUNT(*) FROM anomaly").fetchone()[0]
        conn.close()
        return {
            "satellites": sat_count,
            "tle_records": tle_count,
            "anomalies": anomaly_count,
        }
