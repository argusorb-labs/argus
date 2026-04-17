"""Smoke tests for Starlink constellation tracker."""

import tempfile
import time
import os


def _make_store():
    from services.telemetry.store import StarlinkStore
    db = tempfile.mktemp(suffix=".db")
    return StarlinkStore(db), db


SAMPLE_TLES = [
    {
        "norad_id": 44714, "epoch_jd": 2460400.5,
        "line1": "1 44714C 19074B   26102.88173611  .00004936  00000+0  14452-3 0  1023",
        "line2": "2 44714  53.1552  19.3277 0003480 144.5700 243.9067 15.34670756    14",
        "name": "STARLINK-1008", "inclination": 53.15, "mean_motion": 15.347,
        "eccentricity": 0.000348, "shell_km": 550, "intl_designator": "19074B",
        "launch_group": "19074",
    },
    {
        "norad_id": 44718, "epoch_jd": 2460400.5,
        "line1": "1 44718C 19074F   26102.82965278 -.00009195  00000+0 -27482-3 0  1020",
        "line2": "2 44718  53.1593  19.8308 0003906 134.6049 336.9563 15.34025274    15",
        "name": "STARLINK-1012", "inclination": 53.16, "mean_motion": 15.340,
        "eccentricity": 0.000391, "shell_km": 550, "intl_designator": "19074F",
        "launch_group": "19074",
    },
]


def test_store_crud():
    store, db = _make_store()
    n = store.upsert_tles(SAMPLE_TLES)
    assert n == 2
    assert store.stats["satellites"] == 2
    assert store.stats["tle_records"] == 2

    # Dedup: re-inserting the same TLEs must return 0 new rows (regression
    # guard for the old conn.total_changes bug that counted every iteration).
    n2 = store.upsert_tles(SAMPLE_TLES)
    assert n2 == 0
    assert store.stats["tle_records"] == 2

    latest = store.get_latest_tles()
    assert len(latest) == 2

    sat = store.get_satellite(44714)
    assert sat["name"] == "STARLINK-1008"

    os.unlink(db)


def test_store_anomaly():
    store, db = _make_store()
    store.upsert_tles(SAMPLE_TLES)
    store.insert_anomaly({
        "norad_id": 44714, "anomaly_type": "altitude_change",
        "altitude_before_km": 550, "altitude_after_km": 540,
        "cause": "maneuver", "confidence": 0.8, "classified_by": "delta_v_rule",
    })
    anomalies = store.get_anomalies()
    assert len(anomalies) == 1
    assert anomalies[0]["anomaly_type"] == "altitude_change"
    assert anomalies[0]["cause"] == "maneuver"
    assert anomalies[0]["confidence"] == 0.8
    assert anomalies[0]["classified_by"] == "delta_v_rule"
    os.unlink(db)


def test_store_fetch_log():
    store, db = _make_store()
    store.log_fetch(
        status="ok", http_bytes=12345, parsed_count=100,
        new_tle_count=42, parse_errors=0, duration_ms=830,
        raw_archive_path="data/raw/20260412/celestrak-20260412-120000.tle.gz",
    )
    store.log_fetch(
        status="error", duration_ms=5000,
        error_msg="ReadTimeout: timed out",
    )
    log = store.get_fetch_log()
    assert len(log) == 2
    # Newest first
    assert log[0]["status"] == "error"
    assert log[0]["error_msg"].startswith("ReadTimeout")
    assert log[1]["status"] == "ok"
    assert log[1]["new_tle_count"] == 42
    assert store.stats["fetches"] == 2
    os.unlink(db)


def test_store_migration_idempotent():
    """Running _init_db twice on the same file must not error and must preserve data."""
    from services.telemetry.store import StarlinkStore, SCHEMA_VERSION
    import sqlite3

    store, db = _make_store()
    store.upsert_tles(SAMPLE_TLES)
    store.log_fetch(status="ok", parsed_count=2, new_tle_count=2)

    # Re-open the same DB — _migrate() must be a no-op on already-current schema.
    store2 = StarlinkStore(db)
    assert store2.stats["tle_records"] == 2
    assert store2.stats["fetches"] == 1

    # PRAGMA user_version should equal SCHEMA_VERSION.
    conn = sqlite3.connect(db)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert version == SCHEMA_VERSION

    os.unlink(db)


def test_store_migration_from_v1():
    """A v1 DB (no user_version, no fetch_log, no cause column) must migrate forward."""
    from services.telemetry.store import StarlinkStore, SCHEMA_VERSION
    import sqlite3

    db = tempfile.mktemp(suffix=".db")
    # Hand-build a v1 schema without fetch_log / cause.
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE tle (
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
        CREATE TABLE satellite (
            norad_id INTEGER PRIMARY KEY,
            name TEXT,
            intl_designator TEXT,
            shell_km REAL,
            launch_group TEXT,
            first_seen REAL,
            last_seen REAL,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE anomaly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norad_id INTEGER NOT NULL,
            detected_at REAL NOT NULL,
            anomaly_type TEXT NOT NULL,
            details TEXT,
            altitude_before_km REAL,
            altitude_after_km REAL
        );
    """)
    conn.execute(
        "INSERT INTO tle (norad_id, epoch_jd, fetched_at, line1, line2) VALUES (?, ?, ?, ?, ?)",
        (44714, 2460400.5, time.time(), "line1", "line2"),
    )
    conn.commit()
    conn.close()

    # Opening via StarlinkStore should migrate forward.
    store = StarlinkStore(db)
    assert store.stats["tle_records"] == 1
    assert store.stats["fetches"] == 0

    # fetch_log must exist and be writable.
    store.log_fetch(status="ok", parsed_count=1, new_tle_count=0)
    assert store.stats["fetches"] == 1

    # anomaly.cause column must exist.
    store.insert_anomaly({
        "norad_id": 44714, "anomaly_type": "decay", "cause": "natural_decay",
    })
    anoms = store.get_anomalies()
    assert anoms[0]["cause"] == "natural_decay"

    conn = sqlite3.connect(db)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert version == SCHEMA_VERSION

    os.unlink(db)


def test_propagator():
    from services.telemetry.propagator import Propagator

    prop = Propagator()
    loaded = prop.load_tles(SAMPLE_TLES)
    assert loaded == 2

    positions = prop.propagate_all()
    assert len(positions) == 2
    for p in positions:
        assert -90 <= p["lat"] <= 90
        assert -180 <= p["lon"] <= 180
        assert p["alt_km"] > 100


def test_propagator_performance():
    """Ensure batch propagation is fast enough."""
    from services.telemetry.propagator import Propagator

    prop = Propagator()
    # Create 100 copies with different IDs
    tles = []
    for i in range(100):
        t = dict(SAMPLE_TLES[0])
        t["norad_id"] = 90000 + i
        tles.append(t)
    prop.load_tles(tles)

    t0 = time.perf_counter()
    positions = prop.propagate_all()
    elapsed = time.perf_counter() - t0

    assert len(positions) == 100
    assert elapsed < 1.0  # should be << 1s


def test_tle_parser():
    from services.telemetry.tle_fetcher import parse_tle_text

    text = """STARLINK-1008
1 44714C 19074B   26102.88173611  .00004936  00000+0  14452-3 0  1023
2 44714  53.1552  19.3277 0003480 144.5700 243.9067 15.34670756    14
STARLINK-1012
1 44718C 19074F   26102.82965278 -.00009195  00000+0 -27482-3 0  1020
2 44718  53.1593  19.8308 0003906 134.6049 336.9563 15.34025274    15"""

    tles, errors = parse_tle_text(text)
    assert len(tles) == 2
    assert errors == 0
    assert tles[0]["norad_id"] == 44714
    assert tles[0]["name"] == "STARLINK-1008"
    assert tles[0]["inclination"] > 50


def test_tle_parser_counts_errors():
    """Malformed triplets must be counted, not silently dropped."""
    from services.telemetry.tle_fetcher import parse_tle_text

    text = """STARLINK-1008
1 44714C 19074B   26102.88173611  .00004936  00000+0  14452-3 0  1023
2 44714  53.1552  19.3277 0003480 144.5700 243.9067 15.34670756    14
GARBAGE-SAT
NOT-A-TLE-LINE
ALSO-NOT-A-TLE"""

    tles, errors = parse_tle_text(text)
    assert len(tles) == 1
    assert errors >= 1


# ── Cleaning layer (tle_validator) ──

VALID_TLE_1008_L1 = "1 44714C 19074B   26102.88173611  .00004936  00000+0  14452-3 0  1023"
VALID_TLE_1008_L2 = "2 44714  53.1552  19.3277 0003480 144.5700 243.9067 15.34670756    14"


def test_checksum_algorithm():
    """Standard TLE checksum: digits summed + 1 per '-' minus sign, mod 10."""
    from services.telemetry.tle_validator import _compute_checksum

    assert _compute_checksum(VALID_TLE_1008_L1) == 3
    assert _compute_checksum(VALID_TLE_1008_L2) == 4
    # Line with a real minus sign in the middle
    l1_with_minus = "1 44718C 19074F   26102.82965278 -.00009195  00000+0 -27482-3 0  1020"
    assert _compute_checksum(l1_with_minus) == 0


def test_validate_structure_accepts_good_lines():
    from services.telemetry.tle_validator import validate_tle_structure

    ok, reason = validate_tle_structure(VALID_TLE_1008_L1, VALID_TLE_1008_L2)
    assert ok is True
    assert reason is None


def test_validate_structure_rejects_bad_length():
    from services.telemetry.tle_validator import validate_tle_structure

    short = VALID_TLE_1008_L1[:50]
    ok, reason = validate_tle_structure(short, VALID_TLE_1008_L2)
    assert ok is False
    assert reason == "length"


def test_validate_structure_rejects_corrupted_checksum():
    """Flip the stored checksum digit — everything else valid."""
    from services.telemetry.tle_validator import validate_tle_structure

    corrupted_l1 = VALID_TLE_1008_L1[:68] + "9"  # was '3'
    ok, reason = validate_tle_structure(corrupted_l1, VALID_TLE_1008_L2)
    assert ok is False
    assert reason == "checksum_l1"

    corrupted_l2 = VALID_TLE_1008_L2[:68] + "0"  # was '4'
    ok, reason = validate_tle_structure(VALID_TLE_1008_L1, corrupted_l2)
    assert ok is False
    assert reason == "checksum_l2"


def test_validate_structure_rejects_norad_mismatch():
    """Line 2 NORAD differs from line 1 — catches triplet misalignment."""
    from services.telemetry.tle_validator import validate_tle_structure

    # Same line 2 but NORAD 44714 → 44715 AND checksum rewritten from 4 → 5
    # (single digit change in NORAD shifts the sum by +1, so new checksum = 5).
    mismatch_l2 = "2 44715  53.1552  19.3277 0003480 144.5700 243.9067 15.34670756    15"
    ok, reason = validate_tle_structure(VALID_TLE_1008_L1, mismatch_l2)
    assert ok is False
    assert reason == "norad_mismatch"


def test_validate_physics_accepts_normal_starlink():
    from services.telemetry.tle_validator import validate_tle_physics

    parsed = {
        "mean_motion": 15.34, "eccentricity": 0.000348,
        "inclination": 53.15, "alt_km": 550.0,
    }
    assert validate_tle_physics(parsed) == (True, None)


def test_validate_physics_rejects_hyperbolic_eccentricity():
    from services.telemetry.tle_validator import validate_tle_physics

    parsed = {
        "mean_motion": 15.34, "eccentricity": 1.2,  # escape trajectory
        "inclination": 53.15, "alt_km": 550.0,
    }
    ok, reason = validate_tle_physics(parsed)
    assert ok is False
    assert reason == "eccentricity_range"


def test_validate_physics_rejects_impossible_mean_motion():
    from services.telemetry.tle_validator import validate_tle_physics

    for mm in (0.0, -1.0, 50.0):
        parsed = {
            "mean_motion": mm, "eccentricity": 0.0003,
            "inclination": 53.15, "alt_km": 550.0,
        }
        ok, reason = validate_tle_physics(parsed)
        assert ok is False
        assert reason == "mean_motion_range"


def test_validate_physics_rejects_out_of_range_inclination():
    from services.telemetry.tle_validator import validate_tle_physics

    for incl in (-10.0, 181.0, 360.0):
        parsed = {
            "mean_motion": 15.34, "eccentricity": 0.0003,
            "inclination": incl, "alt_km": 550.0,
        }
        ok, reason = validate_tle_physics(parsed)
        assert ok is False
        assert reason == "inclination_range"


def test_validate_physics_rejects_altitude_inside_earth():
    from services.telemetry.tle_validator import validate_tle_physics

    for alt in (0.0, 100.0, 149.9):
        parsed = {
            "mean_motion": 15.34, "eccentricity": 0.0003,
            "inclination": 53.15, "alt_km": alt,
        }
        ok, reason = validate_tle_physics(parsed)
        assert ok is False
        assert reason == "altitude_range"


def test_parse_tle_float_bstar_format():
    """TLE's compact float format — implied decimal + signed exponent."""
    import math
    from services.telemetry.tle_fetcher import _parse_tle_float

    close = lambda a, b: math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)

    # Unsigned positive mantissa with space sign char
    assert close(_parse_tle_float(" 14452-3"), 0.14452e-3)
    # Explicit negative mantissa
    assert close(_parse_tle_float("-27482-3"), -0.27482e-3)
    # Explicit positive mantissa
    assert close(_parse_tle_float("+12345-5"), 0.12345e-5)
    # Zero in both conventions
    assert _parse_tle_float(" 00000+0") == 0.0
    assert _parse_tle_float(" 00000-0") == 0.0
    # Larger exponents
    assert close(_parse_tle_float(" 99999-9"), 0.99999e-9)
    assert close(_parse_tle_float(" 10000+0"), 0.1)  # 0.10000e0
    # Empty input is 0
    assert _parse_tle_float("") == 0.0
    assert _parse_tle_float("    ") == 0.0


def test_parse_tle_float_plain_decimal_fallback():
    """Plain decimal (like 1st derivative of mean motion) uses float() fallback."""
    import math
    from services.telemetry.tle_fetcher import _parse_tle_float

    close = lambda a, b: math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
    assert close(_parse_tle_float(" .00004936"), 4.936e-5)
    assert close(_parse_tle_float("-.00009195"), -9.195e-5)
    assert _parse_tle_float(".5") == 0.5


def test_parser_extracts_bstar():
    """parse_tle_text must populate bstar in the returned dict."""
    from services.telemetry.tle_fetcher import parse_tle_text

    text = "\n".join([
        "STARLINK-1008", VALID_TLE_1008_L1, VALID_TLE_1008_L2,
        "STARLINK-1012",
        "1 44718C 19074F   26102.82965278 -.00009195  00000+0 -27482-3 0  1020",
        "2 44718  53.1593  19.8308 0003906 134.6049 336.9563 15.34025274    15",
    ])
    tles, errors = parse_tle_text(text)
    assert len(tles) == 2
    assert errors == 0

    # 44714 has B* " 14452-3" → 1.4452e-4 (positive)
    t1 = next(t for t in tles if t["norad_id"] == 44714)
    assert abs(t1["bstar"] - 1.4452e-4) < 1e-10

    # 44718 has B* "-27482-3" → -2.7482e-4 (negative — real NORAD fit)
    t2 = next(t for t in tles if t["norad_id"] == 44718)
    assert abs(t2["bstar"] - (-2.7482e-4)) < 1e-10


def test_store_bstar_roundtrip():
    """bstar must survive upsert_tles → get_satellite_history."""
    store, db = _make_store()
    sample = dict(SAMPLE_TLES[0])
    sample["bstar"] = 1.4452e-4
    store.upsert_tles([sample])

    history = store.get_satellite_history(sample["norad_id"], limit=1)
    assert len(history) == 1
    assert abs(history[0]["bstar"] - 1.4452e-4) < 1e-10
    os.unlink(db)


def test_store_migration_v3_to_v4():
    """A v3 DB (no tle.bstar) must migrate forward and accept bstar writes."""
    from services.telemetry.store import StarlinkStore, SCHEMA_VERSION
    import sqlite3

    db = tempfile.mktemp(suffix=".db")
    # Build a minimal v3-shape DB: tle table without bstar column.
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE tle (
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
        CREATE TABLE satellite (
            norad_id INTEGER PRIMARY KEY,
            name TEXT, intl_designator TEXT, shell_km REAL,
            launch_group TEXT, first_seen REAL, last_seen REAL,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE anomaly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norad_id INTEGER NOT NULL,
            detected_at REAL NOT NULL,
            anomaly_type TEXT NOT NULL,
            details TEXT,
            altitude_before_km REAL,
            altitude_after_km REAL,
            cause TEXT,
            confidence REAL,
            classified_by TEXT,
            source_epoch_jd REAL
        );
        CREATE TABLE fetch_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at REAL NOT NULL,
            status TEXT NOT NULL,
            http_bytes INTEGER, parsed_count INTEGER, new_tle_count INTEGER,
            parse_errors INTEGER, duration_ms INTEGER,
            error_msg TEXT, raw_archive_path TEXT
        );
        PRAGMA user_version = 3;
    """)
    # Seed one legacy tle row (no bstar column yet).
    conn.execute(
        "INSERT INTO tle (norad_id, epoch_jd, fetched_at, line1, line2) VALUES (?, ?, ?, ?, ?)",
        (44714, 2460400.5, time.time(), "line1", "line2"),
    )
    conn.commit()
    conn.close()

    # Opening via StarlinkStore should run the v3 → v4 migration.
    store = StarlinkStore(db)
    assert store.stats["tle_records"] == 1

    conn = sqlite3.connect(db)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION

    # bstar column exists and the legacy row has NULL for it.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tle)").fetchall()]
    assert "bstar" in cols
    row = conn.execute("SELECT bstar FROM tle WHERE norad_id = 44714").fetchone()
    assert row[0] is None
    conn.close()

    # New writes with bstar work end-to-end.
    new_tle = {
        "norad_id": 44714, "epoch_jd": 2460401.0,
        "line1": "x", "line2": "y", "name": "STARLINK-1008",
        "inclination": 53.15, "mean_motion": 15.34, "eccentricity": 0.0003,
        "bstar": 1.4452e-4,
        "shell_km": 550, "intl_designator": "19074B", "launch_group": "19074",
    }
    store.upsert_tles([new_tle])
    history = store.get_satellite_history(44714, limit=5)
    assert any(abs((r.get("bstar") or 0) - 1.4452e-4) < 1e-10 for r in history)

    os.unlink(db)


def test_parse_tle_text_rejects_bad_checksum_end_to_end():
    """A TLE with corrupted checksum must not appear in the parsed results."""
    from services.telemetry.tle_fetcher import parse_tle_text

    # First triplet is valid; second has a flipped checksum on line 1.
    second_l1_bad = "1 44718C 19074F   26102.82965278 -.00009195  00000+0 -27482-3 0  1029"  # was ...0
    second_l2 = "2 44718  53.1593  19.8308 0003906 134.6049 336.9563 15.34025274    15"
    text = "\n".join([
        "STARLINK-1008", VALID_TLE_1008_L1, VALID_TLE_1008_L2,
        "STARLINK-1012", second_l1_bad, second_l2,
    ])

    tles, errors = parse_tle_text(text)
    # Exactly one TLE survives; the corrupted one is rejected.
    assert len(tles) == 1
    assert tles[0]["norad_id"] == 44714
    assert errors >= 1


def test_archive_raw(tmp_path):
    from services.telemetry.tle_fetcher import archive_raw
    import gzip

    path = archive_raw("STARLINK-1008\nline1\nline2\n", raw_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".gz"
    # Round-trip: file must be gzipped and contain the original text.
    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert "STARLINK-1008" in f.read()


def test_orbital_analyzer():
    from services.brain.orbital_analyzer import analyze_tle_pair

    old = {"norad_id": 44714, "mean_motion": 15.34, "eccentricity": 0.0003,
           "epoch_jd": 2460400.0, "inclination": 53.15}
    new = {"norad_id": 44714, "mean_motion": 15.50, "eccentricity": 0.0003,
           "epoch_jd": 2460401.0, "inclination": 53.15}

    anomaly = analyze_tle_pair(old, new)
    assert anomaly is not None
    assert anomaly["anomaly_type"] in ("altitude_change", "deorbiting", "reentry")
    # Step 2: every anomaly must be fully labeled.
    assert anomaly["classified_by"] == "rule_v1"
    assert anomaly["cause"] in ("maneuver_candidate", "natural_decay", "reentry")
    assert 0.0 <= anomaly["confidence"] <= 1.0
    assert anomaly["source_epoch_jd"] == 2460401.0


def test_orbital_analyzer_inclination_shift():
    from services.brain.orbital_analyzer import analyze_tle_pair

    old = {"norad_id": 44714, "mean_motion": 15.34, "eccentricity": 0.0003,
           "epoch_jd": 2460400.0, "inclination": 53.15}
    # 0.2° plane change — far beyond TLE noise.
    new = {"norad_id": 44714, "mean_motion": 15.34, "eccentricity": 0.0003,
           "epoch_jd": 2460401.0, "inclination": 53.35}

    anomaly = analyze_tle_pair(old, new)
    assert anomaly is not None
    assert anomaly["anomaly_type"] == "inclination_shift"
    assert anomaly["cause"] == "maneuver_candidate"


def test_orbital_analyzer_bstar_sign_flip():
    """B* sign flip (boost → coast) must fire when rules 1-5 don't match."""
    from services.brain.orbital_analyzer import analyze_tle_pair

    # Small altitude change (within 10 km threshold), but B* flips sign.
    # Both |B*| > 5e-3 (the raised floor — only large-B* regime sats).
    old = {"norad_id": 44714, "mean_motion": 15.340, "eccentricity": 0.000348,
           "epoch_jd": 2460400.0, "inclination": 53.15,
           "bstar": 1.0e-2}
    new = {"norad_id": 44714, "mean_motion": 15.341, "eccentricity": 0.000350,
           "epoch_jd": 2460401.0, "inclination": 53.15,
           "bstar": -1.2e-2}

    anomaly = analyze_tle_pair(old, new)
    assert anomaly is not None
    assert anomaly["anomaly_type"] == "bstar_sign_flip"
    assert anomaly["cause"] == "maneuver_candidate"
    assert "propulsion mode change" in anomaly["details"]


def test_orbital_analyzer_bstar_sign_flip_below_floor_ignored():
    """B* sign flip with |B*| below the raised floor (5e-3) must NOT fire."""
    from services.brain.orbital_analyzer import analyze_tle_pair

    # Typical Starlink cycling — |B*| = 1e-3 is below the 5e-3 floor.
    # This is normal operational rhythm, not an anomaly.
    old = {"norad_id": 44714, "mean_motion": 15.340, "eccentricity": 0.000348,
           "epoch_jd": 2460400.0, "inclination": 53.15,
           "bstar": 1e-3}
    new = {"norad_id": 44714, "mean_motion": 15.341, "eccentricity": 0.000350,
           "epoch_jd": 2460401.0, "inclination": 53.15,
           "bstar": -1.2e-3}

    assert analyze_tle_pair(old, new) is None


def test_orbital_analyzer_bstar_magnitude_jump():
    """Large B* jump (>200% + >5e-3 absolute) → atmospheric_anomaly."""
    from services.brain.orbital_analyzer import analyze_tle_pair

    # B* triples from 5e-3 to 1.6e-2: ratio = 220% > 200%, |Δ| = 1.1e-2 > 5e-3.
    old = {"norad_id": 44714, "mean_motion": 15.340, "eccentricity": 0.000348,
           "epoch_jd": 2460400.0, "inclination": 53.15,
           "bstar": 5.0e-3}
    new = {"norad_id": 44714, "mean_motion": 15.341, "eccentricity": 0.000350,
           "epoch_jd": 2460401.0, "inclination": 53.15,
           "bstar": 1.6e-2}

    anomaly = analyze_tle_pair(old, new)
    assert anomaly is not None
    assert anomaly["anomaly_type"] == "bstar_anomaly"
    assert anomaly["cause"] == "atmospheric_anomaly"
    assert "%" in anomaly["details"]


def test_orbital_analyzer_bstar_moderate_jump_ignored():
    """B* jump at median range (50% change, ~1e-3 absolute) → no flag."""
    from services.brain.orbital_analyzer import analyze_tle_pair

    # Normal Starlink cycling: 1e-3 to 1.5e-3 is only 50% and |Δ| = 5e-4.
    # Both well below the raised thresholds (200% ratio, 5e-3 abs).
    old = {"norad_id": 44714, "mean_motion": 15.340, "eccentricity": 0.000348,
           "epoch_jd": 2460400.0, "inclination": 53.15,
           "bstar": 1e-3}
    new = {"norad_id": 44714, "mean_motion": 15.341, "eccentricity": 0.000350,
           "epoch_jd": 2460401.0, "inclination": 53.15,
           "bstar": 1.5e-3}

    assert analyze_tle_pair(old, new) is None


def test_orbital_analyzer_altitude_takes_precedence_over_bstar():
    """Rules 1-5 have higher precedence than B* rules 6-7."""
    from services.brain.orbital_analyzer import analyze_tle_pair

    # Both a 20 km altitude change AND a B* sign flip — altitude wins.
    old = {"norad_id": 44714, "mean_motion": 15.34, "eccentricity": 0.0003,
           "epoch_jd": 2460400.0, "inclination": 53.15,
           "bstar": 1e-3}
    new = {"norad_id": 44714, "mean_motion": 15.50, "eccentricity": 0.0003,
           "epoch_jd": 2460401.0, "inclination": 53.15,
           "bstar": -2e-3}

    anomaly = analyze_tle_pair(old, new)
    assert anomaly is not None
    assert anomaly["anomaly_type"] in ("altitude_change", "deorbiting", "reentry")
    # Should NOT be bstar_sign_flip because altitude rule fired first.
    assert anomaly["anomaly_type"] != "bstar_sign_flip"


def test_orbital_analyzer_no_false_positive():
    """Station-keeping noise must not trigger any rule, including B* rules."""
    from services.brain.orbital_analyzer import analyze_tle_pair

    old = {"norad_id": 44714, "mean_motion": 15.340, "eccentricity": 0.000348,
           "epoch_jd": 2460400.0, "inclination": 53.1552,
           "bstar": 1.0e-3}
    new = {"norad_id": 44714, "mean_motion": 15.341, "eccentricity": 0.000350,
           "epoch_jd": 2460401.0, "inclination": 53.1553,
           "bstar": 1.1e-3}  # 10% change, well below 50% threshold

    assert analyze_tle_pair(old, new) is None


def test_anomaly_label_semantics():
    """anomaly table is labeled data — multi-classifier labels must coexist."""
    store, db = _make_store()
    store.upsert_tles(SAMPLE_TLES)

    base = {
        "norad_id": 44714, "anomaly_type": "altitude_change",
        "altitude_before_km": 550, "altitude_after_km": 540,
        "cause": "maneuver_candidate",
        "source_epoch_jd": 2460401.0,
    }

    # 1. rule_v1 labels this event → inserted.
    rule_label = dict(base, confidence=0.7, classified_by="rule_v1")
    assert store.insert_anomaly(rule_label) is True

    # 2. Re-running rule_v1 on the same event → no-op (same classifier, same key).
    assert store.insert_anomaly(rule_label) is False
    assert len(store.get_anomalies()) == 1

    # 3. imm_ukf_v1 labels the SAME event → NEW row. This is the A/B
    #    mechanism: both classifiers' opinions coexist so we can compare.
    imm_label = dict(base, confidence=0.95, classified_by="imm_ukf_v1")
    assert store.insert_anomaly(imm_label) is True
    assert len(store.get_anomalies()) == 2

    # 4. A human reviewer overrides with a different cause → NEW row.
    human_label = dict(base,
        cause="natural_decay", confidence=1.0, classified_by="human:yong")
    assert store.insert_anomaly(human_label) is True
    assert len(store.get_anomalies()) == 3

    # 5. Same classifier, different anomaly_type on the same event → NEW row
    #    (a single classifier may emit multiple labels per transition).
    rule_ecc = dict(rule_label, anomaly_type="eccentricity_change")
    assert store.insert_anomaly(rule_ecc) is True
    assert len(store.get_anomalies()) == 4

    # 6. Legacy rows with NULL source_epoch_jd are treated as distinct
    #    (SQLite NULL semantics) — migration safety.
    legacy = dict(base, classified_by="rule_v1", confidence=0.7)
    legacy.pop("source_epoch_jd")
    assert store.insert_anomaly(legacy) is True
    assert store.insert_anomaly(legacy) is True

    os.unlink(db)


def test_analyze_constellation_idempotent():
    """Running analyze_constellation twice must not duplicate anomalies."""
    from services.brain.orbital_analyzer import analyze_constellation

    store, db = _make_store()

    # Two TLEs for the same sat: epoch 0 (550 km shell) and epoch 1 (raised by ~15 km).
    old = {
        "norad_id": 44714, "epoch_jd": 2460400.0,
        "line1": "x", "line2": "y", "name": "STARLINK-1008",
        "mean_motion": 15.34, "inclination": 53.15, "eccentricity": 0.0003,
        "shell_km": 550, "intl_designator": "19074B", "launch_group": "19074",
    }
    new = dict(old, epoch_jd=2460401.0, mean_motion=15.10)  # raise orbit

    store.upsert_tles([old, new])

    first_run = analyze_constellation(store)
    assert len(first_run) == 1
    assert first_run[0]["cause"] == "maneuver_candidate"
    assert first_run[0]["classified_by"] == "rule_v1"

    # Second run on unchanged history → zero new anomalies.
    second_run = analyze_constellation(store)
    assert second_run == []
    assert len(store.get_anomalies()) == 1

    os.unlink(db)


def test_store_inventory_queries():
    from services.telemetry.store import StarlinkStore
    import sqlite3

    store, db = _make_store()
    store.upsert_tles(SAMPLE_TLES)

    # Both satellites should have been first_seen just now.
    new_sats = store.get_new_satellites(since_ts=time.time() - 60)
    assert len(new_sats) == 2

    # Nothing should be stale yet (both last_seen is fresh).
    stale = store.get_stale_satellites(max_age_s=60)
    assert stale == []

    # Backdate one satellite's last_seen to simulate a 5-day gap.
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE satellite SET last_seen = ? WHERE norad_id = ?",
        (time.time() - 5 * 86400, 44714),
    )
    conn.commit()
    conn.close()

    stale = store.get_stale_satellites(max_age_s=3 * 86400)
    assert len(stale) == 1
    assert stale[0]["norad_id"] == 44714

    os.unlink(db)


def test_detect_tle_gaps():
    """Satellites silent >24h must be detected; fresh ones must not."""
    from services.brain.orbital_analyzer import detect_tle_gaps
    import sqlite3

    store, db = _make_store()
    store.upsert_tles(SAMPLE_TLES)
    now = time.time()

    # Backdate one satellite's last_seen to 48h ago → should be flagged.
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE satellite SET last_seen = ? WHERE norad_id = ?",
        (now - 48 * 3600, 44714),
    )
    # Keep the other fresh → should NOT be flagged.
    conn.execute(
        "UPDATE satellite SET last_seen = ? WHERE norad_id = ?",
        (now - 1 * 3600, 44718),
    )
    conn.commit()
    conn.close()

    gaps = detect_tle_gaps(store, max_gap_s=24 * 3600, now_ts=now)
    assert len(gaps) == 1
    assert gaps[0]["norad_id"] == 44714
    assert gaps[0]["gap_hours"] >= 47

    os.unlink(db)


def test_detect_new_neighbors():
    """Recently-appeared objects near a target's orbit must be found."""
    from services.brain.orbital_analyzer import detect_new_neighbors
    import sqlite3

    store, db = _make_store()
    now = time.time()

    # Target satellite — Starlink at 53°, mm 15.34
    target = {
        "norad_id": 44714, "epoch_jd": 2460400.0,
        "line1": "x", "line2": "y", "name": "STARLINK-TARGET",
        "inclination": 53.15, "mean_motion": 15.34,
        "eccentricity": 0.0003, "shell_km": 550,
        "intl_designator": "19074B", "launch_group": "19074",
    }
    # Neighbor with similar orbit — should be detected
    neighbor_similar = {
        "norad_id": 99001, "epoch_jd": 2460401.0,
        "line1": "x", "line2": "y", "name": "DEBRIS-001",
        "inclination": 53.20, "mean_motion": 15.32,
        "eccentricity": 0.002, "shell_km": 550,
        "intl_designator": "19074Z", "launch_group": "19074",
    }
    # Distant object — different inclination, should NOT match
    neighbor_far = {
        "norad_id": 99002, "epoch_jd": 2460401.0,
        "line1": "x", "line2": "y", "name": "UNRELATED",
        "inclination": 97.6, "mean_motion": 15.34,
        "eccentricity": 0.0001, "shell_km": 550,
        "intl_designator": "25112A", "launch_group": "25112",
    }
    store.upsert_tles([target, neighbor_similar, neighbor_far])

    # Backdate target's first_seen to 30 days ago; neighbors first_seen is "now".
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE satellite SET first_seen = ? WHERE norad_id = ?",
        (now - 30 * 86400, 44714),
    )
    conn.commit()
    conn.close()

    # Search for neighbors that appeared in the last 7 days
    neighbors = detect_new_neighbors(
        store, norad_id=44714, since_ts=now - 7 * 86400
    )

    norad_ids = [n["norad_id"] for n in neighbors]
    assert 99001 in norad_ids        # similar orbit → found
    assert 99002 not in norad_ids    # different inclination → excluded
    assert 44714 not in norad_ids    # target itself excluded

    os.unlink(db)


def test_api_imports():
    from services.api.main import app
    assert app.title == "ArgusOrb API"


def test_fetcher_labels_in_same_cycle(monkeypatch, tmp_path):
    """Integration: a fetch cycle must produce rule_v1 labels for fresh TLEs.

    Seeds the store with an old TLE for a satellite at a higher altitude,
    then runs one cycle of run_tle_fetcher with a mocked Celestrak response
    delivering a new TLE at a lower altitude (uses the known-valid sample
    TLE so it passes the cleaning layer). The ~200 km delta is an obvious
    commanded maneuver; rule_v1 must write one label.
    """
    import asyncio
    from services.telemetry import tle_fetcher
    from services.telemetry.store import StarlinkStore

    db = tmp_path / "fetcher.db"
    store = StarlinkStore(str(db))

    # Baseline TLE bypasses validation via direct upsert (its line1/line2
    # strings are never re-parsed). mean_motion 14.50 → alt ~750 km, clearly
    # different from the new TLE's ~550 km → rule_v1 fires.
    baseline = {
        "norad_id": 44714, "epoch_jd": 2460400.0,
        "name": "STARLINK-1008",
        "line1": "x", "line2": "y",
        "inclination": 53.1552, "mean_motion": 14.50,
        "eccentricity": 0.000348,
        "shell_km": 750, "intl_designator": "19074B", "launch_group": "19074",
    }
    store.upsert_tles([baseline])

    # Mocked Celestrak response uses the known-good sample TLE so it passes
    # the cleaning layer end-to-end.
    mocked_text = (
        "STARLINK-1008\n"
        f"{VALID_TLE_1008_L1}\n"
        f"{VALID_TLE_1008_L2}\n"
    )

    async def fake_fetch():
        return mocked_text

    monkeypatch.setattr(tle_fetcher, "fetch_celestrak", fake_fetch)
    monkeypatch.setattr(tle_fetcher, "RAW_DIR", tmp_path / "raw")

    # Short-circuit the infinite loop: sleep raises to break out after one cycle.
    call_count = {"n": 0}

    async def fake_sleep(_):
        call_count["n"] += 1
        raise RuntimeError("stop after one cycle")

    monkeypatch.setattr(tle_fetcher.asyncio, "sleep", fake_sleep)

    async def run_once():
        try:
            await tle_fetcher.run_tle_fetcher(store, interval=1)
        except RuntimeError as e:
            if "stop after one cycle" not in str(e):
                raise

    asyncio.run(run_once())

    # Cycle ran → one fetch logged, one new TLE parsed, one rule_v1 label.
    assert store.stats["fetches"] == 1
    assert store.stats["tle_records"] == 2  # baseline + new
    labels = store.get_anomalies()
    rule_labels = [l for l in labels if l.get("classified_by") == "rule_v1"]
    assert len(rule_labels) == 1
    assert rule_labels[0]["cause"] == "maneuver_candidate"
    assert rule_labels[0]["norad_id"] == 44714
