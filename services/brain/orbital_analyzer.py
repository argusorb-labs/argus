"""Orbital Analyzer — detects anomalies from consecutive TLE updates.

Compares altitude, mean motion, and eccentricity between a satellite's
two most recent TLEs to classify orbital events.
"""

from __future__ import annotations

import math
import time

MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH_KM = 6371.0


def mean_motion_to_alt_km(mm: float) -> float:
    """Convert mean motion (rev/day) to altitude (km)."""
    if mm <= 0:
        return 0
    n = mm * 2 * math.pi / 86400
    a = (MU_EARTH / (n * n)) ** (1 / 3)
    return a / 1000 - R_EARTH_KM


def analyze_tle_pair(old: dict, new: dict) -> dict | None:
    """Compare two TLEs for the same satellite. Returns anomaly dict or None.

    Args:
        old: Earlier TLE dict with mean_motion, eccentricity, epoch_jd
        new: Later TLE dict

    Returns:
        Anomaly dict if detected, None if nominal.
    """
    norad_id = new["norad_id"]

    mm_old = old.get("mean_motion", 0)
    mm_new = new.get("mean_motion", 0)
    if mm_old <= 0 or mm_new <= 0:
        return None

    alt_old = mean_motion_to_alt_km(mm_old)
    alt_new = mean_motion_to_alt_km(mm_new)
    delta_alt = alt_new - alt_old

    ecc_old = old.get("eccentricity", 0)
    ecc_new = new.get("eccentricity", 0)
    delta_ecc = abs(ecc_new - ecc_old)

    # Time between TLEs (days)
    dt_days = new.get("epoch_jd", 0) - old.get("epoch_jd", 0)
    if dt_days <= 0:
        return None

    # ── Classification ──

    # Imminent reentry
    if alt_new < 250:
        return {
            "norad_id": norad_id,
            "anomaly_type": "reentry",
            "details": f"Altitude {alt_new:.0f} km — imminent reentry",
            "altitude_before_km": alt_old,
            "altitude_after_km": alt_new,
        }

    # Sustained descent (deorbiting)
    if delta_alt < -5 and alt_new < 400:
        return {
            "norad_id": norad_id,
            "anomaly_type": "deorbiting",
            "details": f"Altitude dropped {abs(delta_alt):.1f} km in {dt_days:.1f} days ({alt_old:.0f}→{alt_new:.0f} km)",
            "altitude_before_km": alt_old,
            "altitude_after_km": alt_new,
        }

    # Large altitude change (maneuver)
    if abs(delta_alt) > 10:
        direction = "raised" if delta_alt > 0 else "lowered"
        return {
            "norad_id": norad_id,
            "anomaly_type": "altitude_change",
            "details": f"Orbit {direction} by {abs(delta_alt):.1f} km in {dt_days:.1f} days ({alt_old:.0f}→{alt_new:.0f} km)",
            "altitude_before_km": alt_old,
            "altitude_after_km": alt_new,
        }

    # Large eccentricity change
    if delta_ecc > 0.01:
        return {
            "norad_id": norad_id,
            "anomaly_type": "eccentricity_change",
            "details": f"Eccentricity changed by {delta_ecc:.4f} ({ecc_old:.4f}→{ecc_new:.4f})",
            "altitude_before_km": alt_old,
            "altitude_after_km": alt_new,
        }

    return None


def analyze_constellation(store) -> list[dict]:
    """Run anomaly detection on all satellites with 2+ TLEs.

    Args:
        store: StarlinkStore instance

    Returns:
        List of anomaly dicts.
    """
    anomalies = []

    # Get all satellites
    latest = store.get_latest_tles()

    for sat in latest:
        norad_id = sat["norad_id"]
        history = store.get_satellite_history(norad_id, limit=2)

        if len(history) < 2:
            continue

        # history[0] is newest, history[1] is previous
        new_tle = history[0]
        old_tle = history[1]

        anomaly = analyze_tle_pair(old_tle, new_tle)
        if anomaly:
            anomaly["detected_at"] = time.time()
            store.insert_anomaly(anomaly)
            anomalies.append(anomaly)

    return anomalies
