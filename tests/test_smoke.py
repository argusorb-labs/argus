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
    assert n >= 1
    assert store.stats["satellites"] == 2
    assert store.stats["tle_records"] == 2

    # Dedup
    n2 = store.upsert_tles(SAMPLE_TLES)
    assert store.stats["tle_records"] == 2  # no new rows

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
    })
    anomalies = store.get_anomalies()
    assert len(anomalies) == 1
    assert anomalies[0]["anomaly_type"] == "altitude_change"
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

    tles = parse_tle_text(text)
    assert len(tles) == 2
    assert tles[0]["norad_id"] == 44714
    assert tles[0]["name"] == "STARLINK-1008"
    assert tles[0]["inclination"] > 50


def test_orbital_analyzer():
    from services.brain.orbital_analyzer import analyze_tle_pair

    old = {"norad_id": 44714, "mean_motion": 15.34, "eccentricity": 0.0003, "epoch_jd": 2460400.0}
    new = {"norad_id": 44714, "mean_motion": 15.50, "eccentricity": 0.0003, "epoch_jd": 2460401.0}

    anomaly = analyze_tle_pair(old, new)
    assert anomaly is not None
    assert anomaly["anomaly_type"] in ("altitude_change", "deorbiting", "reentry")


def test_api_imports():
    from services.api.main import app
    assert app.title == "Selene-Insight API"
