"""Tests for the weekly report builder, renderers, and time utilities."""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone


def _make_store():
    from services.telemetry.store import StarlinkStore
    db = tempfile.mktemp(suffix=".db")
    return StarlinkStore(db), db


# ── Time window helpers ──

def test_iso_week_bounds():
    from services.report.weekly import iso_week_bounds

    # 2026-W15 = Mon 2026-04-06 → Mon 2026-04-13 (half-open)
    start_ts, end_ts = iso_week_bounds(2026, 15)
    start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    assert start.year == 2026 and start.month == 4 and start.day == 6
    assert start.hour == 0 and start.minute == 0
    assert end.day == 13
    assert end_ts - start_ts == 7 * 86400


def test_parse_week_string():
    from services.report.weekly import parse_week_string
    assert parse_week_string("2026-W15") == (2026, 15)
    assert parse_week_string("2025-W01") == (2025, 1)


def test_most_recent_complete_week():
    from services.report.weekly import most_recent_complete_week

    # On Monday 2026-04-13, most recent complete week = 2026-W15 (Apr 6-12).
    monday = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
    assert most_recent_complete_week(monday) == (2026, 15)

    # On Sunday 2026-04-12, current week (W15) hasn't ended → W14.
    sunday = datetime(2026, 4, 12, 20, 0, 0, tzinfo=timezone.utc)
    assert most_recent_complete_week(sunday) == (2026, 14)


# ── build_report ──

def _seed_store_for_week(store, week_start: float):
    """Populate a store with a week's worth of synthetic events."""
    # Two "established" satellites already in the store before the window
    # (so they don't count as new_to_orbit).
    pre_ts = week_start - 10 * 86400
    pre_tles = [
        {
            "norad_id": 44714, "epoch_jd": 2460400.0, "line1": "x", "line2": "y",
            "name": "STARLINK-1008", "inclination": 53.15,
            "mean_motion": 15.34, "eccentricity": 0.0003,
            "shell_km": 550, "intl_designator": "19074B", "launch_group": "19074",
        },
        {
            "norad_id": 44718, "epoch_jd": 2460400.0, "line1": "x", "line2": "y",
            "name": "STARLINK-1012", "inclination": 53.15,
            "mean_motion": 15.34, "eccentricity": 0.0003,
            "shell_km": 550, "intl_designator": "19074F", "launch_group": "19074",
        },
    ]
    store.upsert_tles(pre_tles)
    # Backdate their first_seen and last_seen.
    import sqlite3
    week_end = week_start + 7 * 86400
    conn = sqlite3.connect(store._db_path)
    # 44714: tracked (last_seen just before window close, so it's "fresh").
    conn.execute(
        "UPDATE satellite SET first_seen = ?, last_seen = ? WHERE norad_id = ?",
        (pre_ts, week_end - 3600, 44714),
    )
    # 44718: last_seen is 5 days before window START, so it's departed
    # (>7d since last seen at window end).
    conn.execute(
        "UPDATE satellite SET first_seen = ?, last_seen = ? WHERE norad_id = ?",
        (pre_ts, week_start - 5 * 86400, 44718),
    )
    conn.commit()
    conn.close()

    # A brand new satellite that appears during the window.
    new_tles = [
        {
            "norad_id": 59999, "epoch_jd": 2460402.0, "line1": "x", "line2": "y",
            "name": "STARLINK-31999", "inclination": 53.15,
            "mean_motion": 16.2, "eccentricity": 0.0004,
            "shell_km": 340, "intl_designator": "26042A", "launch_group": "26042",
        },
    ]
    store.upsert_tles(new_tles)
    # upsert_tles stamps first_seen with time.time(), but this test may run
    # long after W15 has ended — force first_seen into the window and
    # last_seen to just before window close so it counts as "tracked."
    conn = sqlite3.connect(store._db_path)
    conn.execute(
        "UPDATE satellite SET first_seen = ?, last_seen = ? WHERE norad_id = ?",
        (week_start + 2 * 86400, week_end - 3600, 59999),
    )
    conn.commit()
    conn.close()

    # rule_v1 labels inside the window
    store.insert_anomaly({
        "norad_id": 44714, "anomaly_type": "altitude_change",
        "cause": "maneuver_candidate", "confidence": 0.85,
        "classified_by": "rule_v1",
        "source_epoch_jd": 2460402.0,
        "altitude_before_km": 550, "altitude_after_km": 562,
        "details": "Orbit raised by 12.0 km",
        "detected_at": week_start + 3 * 86400,
    })
    store.insert_anomaly({
        "norad_id": 44714, "anomaly_type": "inclination_shift",
        "cause": "maneuver_candidate", "confidence": 0.6,
        "classified_by": "rule_v1",
        "source_epoch_jd": 2460403.0,
        "details": "Inclination changed by 0.15°",
        "detected_at": week_start + 4 * 86400,
    })
    store.insert_anomaly({
        "norad_id": 44718, "anomaly_type": "deorbiting",
        "cause": "natural_decay", "confidence": 0.92,
        "classified_by": "rule_v1",
        "source_epoch_jd": 2460404.0,
        "altitude_before_km": 312, "altitude_after_km": 268,
        "details": "Altitude dropped 44 km",
        "detected_at": week_start + 2 * 86400,
    })

    # An imm_ukf_v1 label for the same event as the first rule_v1 one.
    # Must be excluded from the report because we filter classified_by=rule_v1.
    store.insert_anomaly({
        "norad_id": 44714, "anomaly_type": "altitude_change",
        "cause": "natural_decay", "confidence": 0.95,
        "classified_by": "imm_ukf_v1",
        "source_epoch_jd": 2460402.0,
        "detected_at": week_start + 3 * 86400,
    })

    # An out-of-window label — must be excluded.
    store.insert_anomaly({
        "norad_id": 44714, "anomaly_type": "altitude_change",
        "cause": "maneuver_candidate", "confidence": 0.5,
        "classified_by": "rule_v1",
        "source_epoch_jd": 2460395.0,
        "detected_at": week_start - 3 * 86400,
    })

    # Fetch log: 21 successful fetches spread over the week.
    for i in range(21):
        store.log_fetch(
            status="ok", parsed_count=10000, new_tle_count=50, parse_errors=0,
            duration_ms=800,
            fetched_at=week_start + i * (86400 * 7 / 21),
        )


def test_build_report_shape_and_filtering():
    from services.report.weekly import build_report, iso_week_bounds

    store, db = _make_store()
    start_ts, end_ts = iso_week_bounds(2026, 15)
    _seed_store_for_week(store, start_ts)

    report = build_report(store, start_ts, end_ts, iso_week="2026-W15")

    assert report["schema_version"] == 1
    assert report["classified_by"] == "rule_v1"
    assert report["window"]["iso_week"] == "2026-W15"
    assert report["window"]["start_ts"] == start_ts
    assert report["window"]["end_ts"] == end_ts

    # Constellation: only 44714 and 59999 have fresh last_seen inside the window.
    # 44718 was backdated to 5 days before week_start, so it's stale (departed).
    const = report["constellation"]
    assert const["total"] >= 1
    assert "550" in const["shells"] or "340" in const["shells"]

    # New satellites: exactly 1 (59999, which first appeared inside the window).
    assert len(report["new_satellites"]) == 1
    assert report["new_satellites"][0]["norad_id"] == 59999

    # Departed: exactly 1 (44718, last_seen 5 days before week_start = 12 days before end).
    assert len(report["departed_satellites"]) == 1
    assert report["departed_satellites"][0]["norad_id"] == 44718

    # Flagged events: 3 in-window rule_v1 labels. imm_ukf_v1 and out-of-window excluded.
    flags = report["flagged_events"]
    assert flags["total"] == 3
    assert flags["by_cause"]["maneuver_candidate"] == 2
    assert flags["by_cause"]["natural_decay"] == 1
    assert flags["by_anomaly_type"]["altitude_change"] == 1
    assert flags["by_anomaly_type"]["inclination_shift"] == 1
    assert flags["by_anomaly_type"]["deorbiting"] == 1

    # Top by confidence: highest first.
    top = flags["top_by_confidence"]
    assert len(top) == 3
    assert top[0]["confidence"] == 0.92  # natural_decay is the highest
    assert top[0]["cause"] == "natural_decay"

    # Data quality.
    dq = report["data_quality"]
    assert dq["fetch_attempts"] == 21
    assert dq["fetch_successes"] == 21
    assert dq["parse_errors"] == 0
    # Gaps between 21 evenly-spaced fetches in 7 days → ~8h
    assert dq["longest_gap_seconds"] > 0
    assert "h" in dq["longest_gap_human"]

    os.unlink(db)


def test_build_report_classifier_isolation():
    """Labels from a non-authoritative classifier must not leak into counts."""
    from services.report.weekly import build_report, iso_week_bounds

    store, db = _make_store()
    start_ts, end_ts = iso_week_bounds(2026, 15)
    # Only an imm_ukf_v1 label exists in the window.
    store.insert_anomaly({
        "norad_id": 44714, "anomaly_type": "altitude_change",
        "cause": "maneuver_candidate", "confidence": 0.9,
        "classified_by": "imm_ukf_v1",
        "source_epoch_jd": 2460402.0,
        "detected_at": start_ts + 86400,
    })

    report = build_report(store, start_ts, end_ts)
    assert report["flagged_events"]["total"] == 0
    assert report["flagged_events"]["by_cause"] == {}

    os.unlink(db)


# ── Renderers ──

def test_render_markdown_smoke():
    from services.report.weekly import build_report, render_markdown, iso_week_bounds

    store, db = _make_store()
    start_ts, end_ts = iso_week_bounds(2026, 15)
    _seed_store_for_week(store, start_ts)

    report = build_report(store, start_ts, end_ts, iso_week="2026-W15")
    md = render_markdown(report)

    assert "ArgusOrb Weekly" in md
    assert "2026-W15" in md
    assert "Constellation" in md
    assert "New to orbit" in md
    assert "Departed" in md
    assert "Flagged events" in md
    assert "Data quality" in md
    assert "rule_v1" in md
    assert "STARLINK-31999" in md or "59999" in md  # new sat
    assert "STARLINK-1012" in md or "44718" in md  # departed sat
    assert "candidate" in md.lower()  # hedged language present

    os.unlink(db)


def test_render_markdown_editor_notes_override():
    from services.report.weekly import build_report, render_markdown, iso_week_bounds

    store, db = _make_store()
    start_ts, end_ts = iso_week_bounds(2026, 15)
    _seed_store_for_week(store, start_ts)

    report = build_report(store, start_ts, end_ts, iso_week="2026-W15")
    notes = "- Custom editorial observation here\n- Another hand-written point"
    md = render_markdown(report, editor_notes=notes)

    assert "Custom editorial observation here" in md
    assert "Another hand-written point" in md
    # The auto-generated footer must NOT appear when editor notes are provided.
    assert "Auto-generated" not in md


def test_render_json_valid():
    from services.report.weekly import build_report, render_json, iso_week_bounds

    store, db = _make_store()
    start_ts, end_ts = iso_week_bounds(2026, 15)
    _seed_store_for_week(store, start_ts)

    report = build_report(store, start_ts, end_ts, iso_week="2026-W15")
    j = render_json(report)
    parsed = json.loads(j)
    assert parsed["window"]["iso_week"] == "2026-W15"
    assert parsed["schema_version"] == 1
    os.unlink(db)


# ── Deltas ──

def test_compute_deltas_populates_weekly_diff():
    from services.report.weekly import compute_deltas

    previous = {
        "schema_version": 1,
        "constellation": {"total": 10200, "shells": {"550": 4600, "560": 2100}},
        "flagged_events": {
            "by_cause": {"maneuver_candidate": 60, "natural_decay": 8},
            "by_anomaly_type": {"altitude_change": 55},
        },
    }
    current = {
        "schema_version": 1,
        "constellation": {"total": 10208, "shells": {"550": 4612, "560": 2104}},
        "flagged_events": {
            "by_cause": {"maneuver_candidate": 71, "natural_decay": 11, "reentry": 1},
            "by_anomaly_type": {"altitude_change": 62},
        },
    }
    d = compute_deltas(current, previous)
    assert d["constellation_total"] == 8
    assert d["shells"]["550"] == 12
    assert d["shells"]["560"] == 4
    assert d["by_cause"]["maneuver_candidate"] == 11
    assert d["by_cause"]["natural_decay"] == 3
    assert d["by_cause"]["reentry"] == 1
    assert d["by_anomaly_type"]["altitude_change"] == 7


def test_compute_deltas_empty_when_no_previous():
    from services.report.weekly import compute_deltas
    assert compute_deltas({"schema_version": 1}, None) == {}
    assert compute_deltas({"schema_version": 1}, {"schema_version": 999}) == {}


def test_render_markdown_with_deltas():
    from services.report.weekly import build_report, render_markdown, iso_week_bounds

    store, db = _make_store()
    start_ts, end_ts = iso_week_bounds(2026, 15)
    _seed_store_for_week(store, start_ts)
    report = build_report(store, start_ts, end_ts, iso_week="2026-W15")

    previous = {
        "schema_version": 1,
        "constellation": {"total": report["constellation"]["total"] - 5, "shells": {}},
        "flagged_events": {
            "by_cause": {"maneuver_candidate": 1, "natural_decay": 0},
            "by_anomaly_type": {},
        },
    }
    md = render_markdown(report, previous=previous)
    assert "+5" in md  # constellation delta
    assert "+1" in md  # maneuver candidate delta (2 - 1)
    os.unlink(db)


# ── Persistence helper ──

def test_load_previous_report(tmp_path):
    from services.report.weekly import load_previous_report

    # Seed a W14 file; asking for W15 should find it.
    prev = {"schema_version": 1, "window": {"iso_week": "2026-W14"}}
    (tmp_path / "2026-W14.json").write_text(json.dumps(prev))

    loaded = load_previous_report(tmp_path, "2026-W15")
    assert loaded is not None
    assert loaded["window"]["iso_week"] == "2026-W14"

    # W16 has no predecessor file on disk.
    assert load_previous_report(tmp_path, "2099-W01") is None
