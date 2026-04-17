"""Orbital Analyzer — rule-based label producer (ruleset v1).

This module produces labels for the anomaly table. The anomaly table is
the labeled dataset — the moat — not a detection log. Every row this
module writes is one opinion from `rule_v1` about one TLE transition.
Future classifiers (`imm_ukf_v1`, human review) will write their own
labels for the same events; all labels coexist by design so they can be
A/B'd against each other.

This classifier is deliberately a "dumb" threshold model — it exists to
(a) seed the labeled history so the future IMM-UKF classifier has a
baseline to validate against, and (b) give the weekly report something
concrete to count.

Ruleset v1 (in precedence order):
  1. reentry              — alt < REENTRY_ALT_KM
  2. natural_decay        — alt trending down AND alt < LOW_ORBIT_ALT_KM
  3. inclination_shift    — |Δincl| > INCLINATION_DELTA_DEG (rare, expensive)
  4. maneuver_candidate   — |Δalt| > MANEUVER_ALT_DELTA_KM
  5. eccentricity_change  — |Δecc| > ECC_DELTA
  6. bstar_sign_flip      — B* sign changed (propulsion mode switch)
  7. bstar_magnitude_jump — |ΔB*| large (atmospheric anomaly or drag model shift)

Every anomaly row is tagged with:
  - cause           — one of {reentry, natural_decay, maneuver_candidate}
  - classified_by   — "rule_v1" (bump when the ruleset changes)
  - confidence      — heuristic [0, 1] based on how far above threshold
  - source_epoch_jd — epoch of the NEW TLE (dedup key for reruns)

When the IMM-UKF classifier lands, it will write rows with
classified_by="imm_ukf_v1" and we can A/B the two tracks.
"""

from __future__ import annotations

import math
import time

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH_KM = 6371.0

# ── Thresholds (ruleset v1) ──
REENTRY_ALT_KM = 250.0        # below this = physical atmosphere, days to reentry
LOW_ORBIT_ALT_KM = 400.0      # decay rule only applies below this
DECAY_DELTA_KM = 5.0          # persistent drop > this in LEO = decaying
MANEUVER_ALT_DELTA_KM = 10.0  # Starlink station-keeping is << 1 km, so 10 km is clearly commanded
INCLINATION_DELTA_DEG = 0.1   # TLE noise is ~0.01°, real plane change is very expensive
ECC_DELTA = 0.01              # eccentricity shift indicating burn

# B* rules — these fire only when rules 1-5 didn't already flag the pair,
# so they catch the "nothing obvious changed in altitude/ecc/incl but the
# drag model shifted" events. Sign flips signal propulsion mode changes
# (boost → coast or reverse). Magnitude jumps without sign change suggest
# atmospheric density anomalies or attitude changes that alter cross-section.
# B* thresholds — tuned against real Starlink data (2026-04-16 backfill).
# Initial thresholds (1e-5 floor, 50% ratio, 5e-4 abs) produced 85k labels
# across 10k sats — B* sign flips are NORMAL operational rhythm for an
# actively-thrust constellation, not anomalies. Raised to catch only the
# outlier events: simultaneous sign flip + large magnitude change, or
# magnitude jumps well above the fleet median (~1e-3).
BSTAR_SIGN_FLIP_FLOOR = 5e-3   # both |old| and |new| must be well above median (large-B* regime)
BSTAR_JUMP_RATIO = 2.0         # relative change > 200% (not 50% — normal cycling is ~50%)
BSTAR_JUMP_ABS_MIN = 5e-3      # absolute change must also be large (10× original threshold)


def mean_motion_to_alt_km(mm: float) -> float:
    """Convert mean motion (rev/day) to altitude (km)."""
    if mm <= 0:
        return 0
    n = mm * 2 * math.pi / 86400
    a = (MU_EARTH / (n * n)) ** (1 / 3)
    return a / 1000 - R_EARTH_KM


def _confidence(delta: float, threshold: float, cap: float = 3.0) -> float:
    """Heuristic confidence: how many thresholds above the floor we are, clamped."""
    if threshold <= 0:
        return 0.5
    ratio = abs(delta) / threshold
    return min(1.0, ratio / cap)


def analyze_tle_pair(old: dict, new: dict) -> dict | None:
    """Compare two TLEs for the same satellite. Returns anomaly dict or None.

    Args:
        old: Earlier TLE dict with mean_motion, eccentricity, epoch_jd, inclination
        new: Later TLE dict

    Returns:
        Anomaly dict with cause/confidence/classified_by/source_epoch_jd if
        detected, else None.
    """
    norad_id = new["norad_id"]

    mm_old = old.get("mean_motion", 0)
    mm_new = new.get("mean_motion", 0)
    if mm_old <= 0 or mm_new <= 0:
        return None

    alt_old = mean_motion_to_alt_km(mm_old)
    alt_new = mean_motion_to_alt_km(mm_new)
    delta_alt = alt_new - alt_old

    ecc_old = old.get("eccentricity", 0) or 0
    ecc_new = new.get("eccentricity", 0) or 0
    delta_ecc = abs(ecc_new - ecc_old)

    incl_old = old.get("inclination", 0) or 0
    incl_new = new.get("inclination", 0) or 0
    delta_incl = abs(incl_new - incl_old)

    dt_days = new.get("epoch_jd", 0) - old.get("epoch_jd", 0)
    if dt_days <= 0:
        return None

    base = {
        "norad_id": norad_id,
        "altitude_before_km": alt_old,
        "altitude_after_km": alt_new,
        "source_epoch_jd": new.get("epoch_jd"),
        "classified_by": "rule_v1",
    }

    # 1. Imminent reentry
    if alt_new < REENTRY_ALT_KM:
        return {
            **base,
            "anomaly_type": "reentry",
            "cause": "reentry",
            "confidence": 1.0,
            "details": f"Altitude {alt_new:.0f} km — imminent reentry",
        }

    # 2. Natural decay (sustained descent in LEO)
    if delta_alt < -DECAY_DELTA_KM and alt_new < LOW_ORBIT_ALT_KM:
        return {
            **base,
            "anomaly_type": "deorbiting",
            "cause": "natural_decay",
            "confidence": _confidence(delta_alt, DECAY_DELTA_KM),
            "details": (
                f"Altitude dropped {abs(delta_alt):.1f} km in {dt_days:.1f} days "
                f"({alt_old:.0f}→{alt_new:.0f} km)"
            ),
        }

    # 3. Inclination shift (very rare — plane change is expensive)
    if delta_incl > INCLINATION_DELTA_DEG:
        return {
            **base,
            "anomaly_type": "inclination_shift",
            "cause": "maneuver_candidate",
            "confidence": _confidence(delta_incl, INCLINATION_DELTA_DEG),
            "details": (
                f"Inclination changed by {delta_incl:.3f}° in {dt_days:.1f} days "
                f"({incl_old:.3f}°→{incl_new:.3f}°)"
            ),
        }

    # 4. Large altitude change (station-keeping or commanded maneuver)
    if abs(delta_alt) > MANEUVER_ALT_DELTA_KM:
        direction = "raised" if delta_alt > 0 else "lowered"
        return {
            **base,
            "anomaly_type": "altitude_change",
            "cause": "maneuver_candidate",
            "confidence": _confidence(delta_alt, MANEUVER_ALT_DELTA_KM),
            "details": (
                f"Orbit {direction} by {abs(delta_alt):.1f} km in {dt_days:.1f} days "
                f"({alt_old:.0f}→{alt_new:.0f} km)"
            ),
        }

    # 5. Eccentricity change
    if delta_ecc > ECC_DELTA:
        return {
            **base,
            "anomaly_type": "eccentricity_change",
            "cause": "maneuver_candidate",
            "confidence": _confidence(delta_ecc, ECC_DELTA),
            "details": (
                f"Eccentricity changed by {delta_ecc:.4f} "
                f"({ecc_old:.4f}→{ecc_new:.4f})"
            ),
        }

    # ── B* rules (fire only if rules 1-5 didn't match) ──

    bstar_old = old.get("bstar")
    bstar_new = new.get("bstar")

    if bstar_old is not None and bstar_new is not None:
        abs_old = abs(bstar_old)
        abs_new = abs(bstar_new)
        delta_bstar = abs(bstar_new - bstar_old)
        denom = max(abs_old, BSTAR_SIGN_FLIP_FLOOR)

        # 6. B* sign flip — propulsion mode switch (boost ↔ coast)
        if (abs_old > BSTAR_SIGN_FLIP_FLOOR
                and abs_new > BSTAR_SIGN_FLIP_FLOOR
                and (bstar_old > 0) != (bstar_new > 0)):
            return {
                **base,
                "anomaly_type": "bstar_sign_flip",
                "cause": "maneuver_candidate",
                "confidence": _confidence(delta_bstar, denom),
                "details": (
                    f"B* sign flipped ({bstar_old:+.3e}→{bstar_new:+.3e}) "
                    f"in {dt_days:.1f} days — propulsion mode change"
                ),
            }

        # 7. B* magnitude jump — atmospheric anomaly or drag model shift
        if (delta_bstar > BSTAR_JUMP_ABS_MIN
                and delta_bstar / denom > BSTAR_JUMP_RATIO):
            return {
                **base,
                "anomaly_type": "bstar_anomaly",
                "cause": "atmospheric_anomaly",
                "confidence": _confidence(delta_bstar / denom, BSTAR_JUMP_RATIO),
                "details": (
                    f"B* jumped by {delta_bstar:.3e} "
                    f"({bstar_old:+.3e}→{bstar_new:+.3e}, "
                    f"{100*delta_bstar/denom:.0f}% change) in {dt_days:.1f} days"
                ),
            }

    return None


def analyze_constellation(store, since_ts: float | None = None) -> list[dict]:
    """Produce rule_v1 labels for every satellite's two most recent TLEs.

    Idempotent — the anomaly table's unique index on
    (norad_id, source_epoch_jd, anomaly_type, classified_by) means
    re-running rule_v1 on the same TLE pairs inserts zero rows. A different
    classifier (imm_ukf_v1, human) writing its own label for the same event
    IS a new row, by design. Returns only the labels that were NEWLY
    written this call (drives weekly-report "detected this week").

    Args:
        store: StarlinkStore instance
        since_ts: If given, only examine satellites whose last_seen >= since_ts
                  (cheap prefilter for incremental runs after a fetch).
    """
    new_anomalies: list[dict] = []
    now = time.time()

    latest = store.get_latest_tles()
    for sat in latest:
        norad_id = sat["norad_id"]

        if since_ts is not None:
            meta = store.get_satellite(norad_id)
            if not meta or (meta.get("last_seen") or 0) < since_ts:
                continue

        history = store.get_satellite_history(norad_id, limit=2)
        if len(history) < 2:
            continue

        new_tle, old_tle = history[0], history[1]
        anomaly = analyze_tle_pair(old_tle, new_tle)
        if anomaly is None:
            continue

        anomaly["detected_at"] = now
        if store.insert_anomaly(anomaly):
            new_anomalies.append(anomaly)

    return new_anomalies


# ── Event detection (non-label, query-based) ──

TLE_GAP_THRESHOLD_S = 24 * 3600  # 24h without TLE update = alert


def detect_tle_gaps(
    store, max_gap_s: float = TLE_GAP_THRESHOLD_S, now_ts: float | None = None
) -> list[dict]:
    """Find satellites with TLE gaps exceeding max_gap_s.

    This is the "satellite went silent" detector. A 24h gap in Starlink
    TLE updates is abnormal and may indicate: satellite failure, breakup
    event, NORAD tracking loss, or catalog maintenance. The gap itself is
    often detectable BEFORE the official announcement.

    Returns enriched dicts (not written to anomaly table — gaps are
    transient events, not permanent labels on TLE transitions).
    """
    return store.get_satellites_with_gap(max_gap_s=max_gap_s, now_ts=now_ts)


def detect_new_neighbors(
    store,
    norad_id: int,
    since_ts: float | None = None,
    incl_tol: float = 0.5,
    mm_tol: float = 0.2,
) -> list[dict]:
    """Find recently-cataloged objects near a given satellite's orbit.

    After a debris event, NORAD catalogs debris pieces under new NORAD IDs
    with similar orbital elements. This function looks for satellites that
    appeared after since_ts in the same orbital neighborhood as norad_id.

    If N new objects appear in the neighborhood of a satellite that went
    silent → strong evidence of a breakup.
    """
    sat = store.get_satellite(norad_id)
    if not sat:
        return []

    # Get target's orbital params from latest TLE
    history = store.get_satellite_history(norad_id, limit=1)
    if not history:
        return []

    target_incl = history[0].get("inclination", 0)
    target_mm = history[0].get("mean_motion", 0)
    if not target_incl or not target_mm:
        return []

    if since_ts is None:
        since_ts = (sat.get("last_seen") or time.time()) - 7 * 86400

    neighbors = store.find_new_neighbors(
        target_incl=target_incl,
        target_mm=target_mm,
        since_ts=since_ts,
        incl_tol=incl_tol,
        mm_tol=mm_tol,
    )
    # Exclude the target itself
    return [n for n in neighbors if n["norad_id"] != norad_id]


def label_full_history(
    store, *, batch_log_interval: int = 500, max_history: int = 1000
) -> int:
    """Run rule_v1 over ALL consecutive TLE pairs for EVERY satellite.

    This is the backfill counterpart to analyze_constellation's incremental
    (latest-2-only) pass. It walks each satellite's full stored history and
    labels every adjacent pair. Idempotent via the unique index — previously
    labeled pairs are silently skipped.

    Intended to be run once after deploying new rules, so the anomaly table
    reflects the full historical reach of the current ruleset rather than only
    events that happened after deployment.

    Returns total new labels written.
    """
    total_new = 0
    now = time.time()
    all_sats = store.get_latest_tles()

    for i, sat in enumerate(all_sats):
        norad_id = sat["norad_id"]
        history = store.get_satellite_history(norad_id, limit=max_history)

        for j in range(len(history) - 1):
            new_tle, old_tle = history[j], history[j + 1]
            anomaly = analyze_tle_pair(old_tle, new_tle)
            if anomaly is None:
                continue
            anomaly["detected_at"] = now
            if store.insert_anomaly(anomaly):
                total_new += 1

        if (i + 1) % batch_log_interval == 0:
            print(
                f"[backfill labels] {i+1}/{len(all_sats)} sats, "
                f"{total_new} new labels so far"
            )

    print(
        f"[backfill labels] done: {len(all_sats)} sats scanned, "
        f"{total_new} new labels written"
    )
    return total_new
