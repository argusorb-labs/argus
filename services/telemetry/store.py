"""SQLite-backed persistent store for Starlink TLE data.

Tables:
- tle:       every TLE update, deduplicated by (norad_id, epoch_jd)
- satellite: metadata, shell classification, status
- anomaly:   LABELED DATASET — every row is a label produced by some
             classifier (rule_v1, imm_ukf_v1, human, ...) about a specific
             TLE transition. Multiple classifiers can label the same event;
             comparing their labels is the validation-credibility moat.
- fetch_log: audit trail of every Celestrak fetch attempt

The combination of `tle` (every update stored forever) and `anomaly`
(every classifier opinion stored forever) is the data moat. Schema is
versioned via PRAGMA user_version and migrated forward.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = os.environ.get("ARGUS_DB_PATH", "data/starlink.db")

SCHEMA_VERSION = 6

_SCHEMA_V1 = """
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

_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at REAL NOT NULL,
    status TEXT NOT NULL,
    http_bytes INTEGER,
    parsed_count INTEGER,
    new_tle_count INTEGER,
    parse_errors INTEGER,
    duration_ms INTEGER,
    error_msg TEXT,
    raw_archive_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_fetch_log_time ON fetch_log(fetched_at DESC);
"""

_ANOMALY_V2_COLUMNS = [
    ("cause", "TEXT"),
    ("confidence", "REAL"),
    ("classified_by", "TEXT"),
]

_ANOMALY_V3_COLUMNS = [
    ("source_epoch_jd", "REAL"),
]

# Schema v4 — add B* drag term to tle. The B* coefficient is the only
# TLE field that lets classifiers distinguish "atmospheric anomaly" from
# "commanded maneuver", so persisting it is a prerequisite for any IMM-UKF
# work. NULL for legacy rows pre-v4 — they can be backfilled by re-parsing
# data/raw/*.tle.gz if needed.
_TLE_V4_COLUMNS = [
    ("bstar", "REAL"),
]

# Schema v5 — supplemental GP table. Stores operator-sourced TLEs (e.g.,
# Planet Labs precise-tracking-derived GP from Celestrak). Parallel to the
# tle table but keyed by (norad_id, epoch_jd, source) so data from multiple
# operators coexists. Used for precision calibration: comparing standard
# NORAD TLEs against operator-quality TLEs reveals SGP4's systematic errors.
_SCHEMA_V5 = """
CREATE TABLE IF NOT EXISTS supplemental_gp (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    norad_id INTEGER NOT NULL,
    epoch_jd REAL NOT NULL,
    fetched_at REAL NOT NULL,
    source TEXT NOT NULL,
    line1 TEXT NOT NULL,
    line2 TEXT NOT NULL,
    inclination REAL,
    mean_motion REAL,
    eccentricity REAL,
    bstar REAL,
    UNIQUE(norad_id, epoch_jd, source)
);
CREATE INDEX IF NOT EXISTS idx_supgp_norad ON supplemental_gp(norad_id, epoch_jd DESC);
CREATE INDEX IF NOT EXISTS idx_supgp_source ON supplemental_gp(source, fetched_at DESC);
"""

# Uniqueness semantics: one classifier can emit at most one label of a given
# anomaly_type per TLE transition. Different classifiers (rule_v1 vs
# imm_ukf_v1 vs human) can all label the same event — that's the A/B /
# validation-credibility mechanism. NULL source_epoch_jd rows (legacy) are
# treated as distinct by SQLite, so this is migration-safe.
_SCHEMA_V3 = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_anomaly_unique
    ON anomaly(norad_id, source_epoch_jd, anomaly_type, classified_by);
"""


# Schema v6 — SatNOGS RF observations for gap cross-validation.
_SCHEMA_V6 = """
CREATE TABLE IF NOT EXISTS satnogs_observation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL,
    norad_id INTEGER NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT,
    ground_station INTEGER,
    vetted_status TEXT,
    frequency_hz INTEGER,
    has_waterfall INTEGER DEFAULT 0,
    has_audio INTEGER DEFAULT 0,
    fetched_at REAL NOT NULL,
    UNIQUE(observation_id)
);
CREATE INDEX IF NOT EXISTS idx_satnogs_norad ON satnogs_observation(norad_id, start_ts DESC);
CREATE INDEX IF NOT EXISTS idx_satnogs_status ON satnogs_observation(vetted_status);
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
            try:
                self._migrate(conn)
            finally:
                conn.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]

        if version < 1:
            conn.executescript(_SCHEMA_V1)

        if version < 2:
            conn.executescript(_SCHEMA_V2)
            for col, col_type in _ANOMALY_V2_COLUMNS:
                try:
                    conn.execute(f"ALTER TABLE anomaly ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # column already exists

        if version < 3:
            for col, col_type in _ANOMALY_V3_COLUMNS:
                try:
                    conn.execute(f"ALTER TABLE anomaly ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass
            conn.executescript(_SCHEMA_V3)

        if version < 4:
            for col, col_type in _TLE_V4_COLUMNS:
                try:
                    conn.execute(f"ALTER TABLE tle ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass

        if version < 5:
            conn.executescript(_SCHEMA_V5)

        if version < 6:
            conn.executescript(_SCHEMA_V6)

        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

    # ── TLE operations ──

    def upsert_tles(self, tles: list[dict]) -> int:
        """Insert TLEs, skipping duplicates. Returns count of genuinely new rows."""
        now = time.time()
        new_count = 0
        with self._lock:
            conn = self._get_conn()
            for t in tles:
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO tle
                       (norad_id, epoch_jd, fetched_at, line1, line2,
                        inclination, mean_motion, eccentricity, bstar)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        t["norad_id"], t["epoch_jd"], now,
                        t["line1"], t["line2"],
                        t.get("inclination"), t.get("mean_motion"),
                        t.get("eccentricity"), t.get("bstar"),
                    ),
                )
                if cursor.rowcount > 0:
                    new_count += 1

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

    def insert_anomaly(self, anomaly: dict) -> bool:
        """Write a label into the anomaly table. Returns True if a new row was inserted.

        This is the labeled dataset, not a detection log. Each row is one
        classifier's opinion about one TLE transition. Dedupe is on
        (norad_id, source_epoch_jd, anomaly_type, classified_by) — re-running
        the SAME classifier on the SAME TLE pair is a no-op, but a different
        classifier labeling the same event inserts a new row (by design).
        """
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """INSERT OR IGNORE INTO anomaly
                   (norad_id, detected_at, anomaly_type, details,
                    altitude_before_km, altitude_after_km,
                    cause, confidence, classified_by, source_epoch_jd)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    anomaly["norad_id"],
                    anomaly.get("detected_at", time.time()),
                    anomaly["anomaly_type"],
                    anomaly.get("details", ""),
                    anomaly.get("altitude_before_km"),
                    anomaly.get("altitude_after_km"),
                    anomaly.get("cause"),
                    anomaly.get("confidence"),
                    anomaly.get("classified_by"),
                    anomaly.get("source_epoch_jd"),
                ),
            )
            inserted = cursor.rowcount > 0
            conn.commit()
            conn.close()
        return inserted

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

    def get_anomalies_in_window(
        self,
        start_ts: float,
        end_ts: float,
        classified_by: str | None = None,
    ) -> list[dict]:
        """Labels with detected_at in [start_ts, end_ts).

        If classified_by is given, restrict to labels from that one
        classifier — the weekly report MUST pass this (otherwise rule_v1
        and imm_ukf_v1 labels for the same event would both be counted).
        """
        conn = self._get_conn()
        if classified_by is None:
            rows = conn.execute(
                """SELECT a.*, s.name FROM anomaly a
                   LEFT JOIN satellite s ON a.norad_id = s.norad_id
                   WHERE a.detected_at >= ? AND a.detected_at < ?
                   ORDER BY a.detected_at DESC""",
                (start_ts, end_ts),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT a.*, s.name FROM anomaly a
                   LEFT JOIN satellite s ON a.norad_id = s.norad_id
                   WHERE a.detected_at >= ? AND a.detected_at < ?
                     AND a.classified_by = ?
                   ORDER BY a.detected_at DESC""",
                (start_ts, end_ts, classified_by),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Inventory queries (for weekly report) ──

    def get_new_satellites(self, since_ts: float) -> list[dict]:
        """Satellites first seen at or after since_ts (new launches / deployments)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM satellite
               WHERE first_seen >= ?
               ORDER BY first_seen DESC""",
            (since_ts,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_stale_satellites(
        self, max_age_s: float, now_ts: float | None = None
    ) -> list[dict]:
        """Satellites whose last_seen is older than (now_ts - max_age_s).

        A Starlink that hasn't shown up in Celestrak for several days is
        likely decommissioned, decayed, or renamed.
        """
        cutoff = (now_ts if now_ts is not None else time.time()) - max_age_s
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM satellite
               WHERE last_seen < ?
               ORDER BY last_seen ASC""",
            (cutoff,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Fetch log ──

    def log_fetch(
        self,
        *,
        status: str,
        http_bytes: int = 0,
        parsed_count: int = 0,
        new_tle_count: int = 0,
        parse_errors: int = 0,
        duration_ms: int = 0,
        error_msg: str | None = None,
        raw_archive_path: str | None = None,
        fetched_at: float | None = None,
    ) -> int:
        """Record a Celestrak fetch attempt. Returns the fetch_log row id."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """INSERT INTO fetch_log
                   (fetched_at, status, http_bytes, parsed_count, new_tle_count,
                    parse_errors, duration_ms, error_msg, raw_archive_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fetched_at if fetched_at is not None else time.time(),
                    status, http_bytes, parsed_count, new_tle_count,
                    parse_errors, duration_ms, error_msg, raw_archive_path,
                ),
            )
            row_id = cursor.lastrowid
            conn.commit()
            conn.close()
        return row_id

    def get_fetch_log(self, limit: int = 50) -> list[dict]:
        """Get recent fetch attempts, newest first."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM fetch_log ORDER BY fetched_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_fetch_log_in_window(
        self, start_ts: float, end_ts: float
    ) -> list[dict]:
        """Fetch attempts with fetched_at in [start_ts, end_ts), oldest first.

        Oldest-first ordering makes gap analysis trivial for the caller.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM fetch_log
               WHERE fetched_at >= ? AND fetched_at < ?
               ORDER BY fetched_at ASC""",
            (start_ts, end_ts),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def count_fresh_by_shell(
        self, as_of_ts: float, freshness_s: float = 86400
    ) -> dict[float, int]:
        """Shell population among satellites whose last_seen is fresh.

        "Currently tracked" = last_seen within freshness_s of as_of_ts.
        Satellites that haven't shown up in Celestrak recently are excluded
        (they'll appear in get_stale_satellites instead).
        """
        cutoff = as_of_ts - freshness_s
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT shell_km, COUNT(*) as n FROM satellite
               WHERE last_seen >= ?
               GROUP BY shell_km""",
            (cutoff,),
        ).fetchall()
        conn.close()
        return {(r["shell_km"] or 0.0): r["n"] for r in rows}

    # ── Batch / launch group queries ──

    def get_batch_siblings(self, intl_designator_prefix: str) -> list[dict]:
        """All satellites from the same launch group, with latest orbital elements.

        Used by the investigator to compare a flagged satellite against its
        siblings — if one satellite's eccentricity is 14× the batch average,
        that's strong evidence of a debris event.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT s.norad_id, s.name, s.shell_km, s.intl_designator,
                      s.first_seen, s.last_seen, s.status,
                      t.mean_motion, t.eccentricity, t.inclination, t.bstar, t.epoch_jd
               FROM satellite s
               JOIN tle t ON s.norad_id = t.norad_id
               WHERE s.intl_designator LIKE ? || '%'
                 AND t.epoch_jd = (SELECT MAX(epoch_jd) FROM tle WHERE norad_id = s.norad_id)
               ORDER BY s.norad_id""",
            (intl_designator_prefix,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Event detection queries ──

    def get_satellites_with_gap(
        self, max_gap_s: float, now_ts: float | None = None
    ) -> list[dict]:
        """Satellites whose last TLE is older than max_gap_s.

        Returns enriched dicts with gap_hours. Unlike get_stale_satellites
        (which is for the weekly report's "departed" section with 7-day
        threshold), this is for real-time gap alerting (~24h threshold).
        """
        cutoff = (now_ts if now_ts is not None else time.time()) - max_gap_s
        now = now_ts if now_ts is not None else time.time()
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT norad_id, name, shell_km, last_seen, status,
                      inclination, intl_designator
               FROM satellite s
               LEFT JOIN (
                   SELECT norad_id as nid, inclination
                   FROM tle
                   GROUP BY norad_id
                   HAVING MAX(epoch_jd)
               ) t ON s.norad_id = t.nid
               WHERE s.last_seen < ? AND s.last_seen > 0
               ORDER BY s.last_seen ASC""",
            (cutoff,),
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            d = dict(r)
            d["gap_hours"] = round((now - (d.get("last_seen") or 0)) / 3600, 1)
            result.append(d)
        return result

    def find_new_neighbors(
        self,
        target_incl: float,
        target_mm: float,
        since_ts: float,
        incl_tol: float = 0.5,
        mm_tol: float = 0.2,
    ) -> list[dict]:
        """Satellites that first appeared after since_ts with similar orbit.

        Used to detect debris pieces: if a satellite goes silent and new
        catalog entries pop up in the same orbital neighborhood, it's
        likely a breakup event.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT s.norad_id, s.name, s.shell_km, s.first_seen,
                      s.intl_designator, t.inclination, t.mean_motion
               FROM satellite s
               JOIN tle t ON s.norad_id = t.norad_id
               WHERE s.first_seen >= ?
                 AND t.epoch_jd = (
                     SELECT MAX(epoch_jd) FROM tle WHERE norad_id = s.norad_id
                 )
                 AND ABS(t.inclination - ?) < ?
                 AND ABS(t.mean_motion - ?) < ?
               ORDER BY s.first_seen DESC""",
            (since_ts, target_incl, incl_tol, target_mm, mm_tol),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Supplemental GP ──

    def upsert_supgp_tles(self, tles: list[dict], source: str) -> int:
        """Insert supplemental GP TLEs. Returns count of genuinely new rows."""
        now = time.time()
        new_count = 0
        with self._lock:
            conn = self._get_conn()
            for t in tles:
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO supplemental_gp
                       (norad_id, epoch_jd, fetched_at, source, line1, line2,
                        inclination, mean_motion, eccentricity, bstar)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        t["norad_id"], t["epoch_jd"], now, source,
                        t["line1"], t["line2"],
                        t.get("inclination"), t.get("mean_motion"),
                        t.get("eccentricity"), t.get("bstar"),
                    ),
                )
                if cursor.rowcount > 0:
                    new_count += 1
            conn.commit()
            conn.close()
        return new_count

    # ── SatNOGS observations ──

    def upsert_satnogs_observations(self, observations: list[dict]) -> int:
        """Insert SatNOGS observations. Returns count of new rows."""
        now = time.time()
        new_count = 0
        with self._lock:
            conn = self._get_conn()
            for o in observations:
                obs_id = o.get("observation_id")
                if obs_id is None:
                    continue
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO satnogs_observation
                       (observation_id, norad_id, start_ts, end_ts,
                        ground_station, vetted_status, frequency_hz,
                        has_waterfall, has_audio, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        obs_id,
                        o.get("norad_id"),
                        o.get("start_ts", ""),
                        o.get("end_ts", ""),
                        o.get("ground_station"),
                        o.get("vetted_status", "unknown"),
                        o.get("frequency_hz"),
                        1 if o.get("has_waterfall") else 0,
                        1 if o.get("has_audio") else 0,
                        now,
                    ),
                )
                if cursor.rowcount > 0:
                    new_count += 1
            conn.commit()
            conn.close()
        return new_count

    def get_satnogs_observations(
        self, norad_id: int, limit: int = 100
    ) -> list[dict]:
        """Get SatNOGS observations for a satellite, newest first."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM satnogs_observation
               WHERE norad_id = ?
               ORDER BY start_ts DESC LIMIT ?""",
            (norad_id, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_satnogs_stats(self) -> dict:
        """SatNOGS observation counts by status."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT vetted_status, COUNT(*) as n
               FROM satnogs_observation GROUP BY vetted_status"""
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM satnogs_observation"
        ).fetchone()[0]
        conn.close()
        return {
            "total": total,
            "by_status": {r["vetted_status"]: r["n"] for r in rows},
        }

    def get_supgp_stats(self) -> dict:
        """Supplemental GP row counts by source."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source, COUNT(*) as n FROM supplemental_gp GROUP BY source"
        ).fetchall()
        conn.close()
        return {r["source"]: r["n"] for r in rows}

    # ── Stats ──

    @property
    def stats(self) -> dict:
        conn = self._get_conn()
        sat_count = conn.execute("SELECT COUNT(*) FROM satellite").fetchone()[0]
        tle_count = conn.execute("SELECT COUNT(*) FROM tle").fetchone()[0]
        anomaly_count = conn.execute("SELECT COUNT(*) FROM anomaly").fetchone()[0]
        fetch_count = conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]
        supgp_count = conn.execute("SELECT COUNT(*) FROM supplemental_gp").fetchone()[0]
        conn.close()
        return {
            "satellites": sat_count,
            "tle_records": tle_count,
            "anomalies": anomaly_count,
            "fetches": fetch_count,
            "supplemental_gp": supgp_count,
        }
