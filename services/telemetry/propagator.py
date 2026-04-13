"""SGP4 Propagator — computes satellite positions at any timestamp.

Propagates all Starlink TLEs to lat/lon/alt positions.
Caches Satrec objects for performance. ~0.3s for 6000 sats.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone

from sgp4.api import Satrec, WGS72
from sgp4.api import jday

R_EARTH_KM = 6371.0


def tle_to_satrec(line1: str, line2: str) -> Satrec | None:
    """Parse TLE lines into an SGP4 Satrec object."""
    try:
        return Satrec.twoline2rv(line1, line2, WGS72)
    except Exception:
        return None


def propagate_single(sat: Satrec, timestamp: float) -> dict | None:
    """Propagate a single satellite to a Unix timestamp.

    Returns dict with lat, lon, alt_km, x/y/z ECI, or None on error.
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    jd, fr = jday(dt.year, dt.month, dt.day,
                  dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None  # propagation error

    x, y, z = r  # km, ECI
    vx, vy, vz = v  # km/s, ECI

    # Convert ECI to lat/lon/alt
    # Simplified: ignore Earth rotation (adequate for visualization)
    r_mag = math.sqrt(x * x + y * y + z * z)
    if r_mag < 1:
        return None

    lat = math.degrees(math.asin(z / r_mag))
    lon = math.degrees(math.atan2(y, x))

    # Account for Earth rotation (Greenwich Sidereal Time)
    # GMST approximation
    d = jd - 2451545.0 + fr
    gmst = 280.46061837 + 360.98564736629 * d
    lon = (lon - gmst) % 360
    if lon > 180:
        lon -= 360

    alt_km = r_mag - R_EARTH_KM

    return {
        "lat": lat,
        "lon": lon,
        "alt_km": alt_km,
        "x_km": x,
        "y_km": y,
        "z_km": z,
        "vx_kms": vx,
        "vy_kms": vy,
        "vz_kms": vz,
    }


class Propagator:
    """Batch SGP4 propagator with Satrec caching."""

    def __init__(self) -> None:
        self._satrecs: dict[int, Satrec] = {}  # norad_id → Satrec
        self._metadata: dict[int, dict] = {}  # norad_id → {name, shell_km, status}

    def load_tles(self, tles: list[dict]) -> int:
        """Load TLEs into Satrec cache. Returns count loaded."""
        count = 0
        for t in tles:
            norad_id = t["norad_id"]
            sat = tle_to_satrec(t["line1"], t["line2"])
            if sat:
                self._satrecs[norad_id] = sat
                self._metadata[norad_id] = {
                    "name": t.get("name", ""),
                    "shell_km": t.get("shell_km", 0),
                    "status": t.get("status", "active"),
                }
                count += 1
        return count

    def propagate_all(self, timestamp: float | None = None) -> list[dict]:
        """Propagate all cached satellites to a timestamp.

        Returns list of {norad_id, name, lat, lon, alt_km, shell_km, status}.
        """
        if timestamp is None:
            timestamp = time.time()

        results = []
        for norad_id, sat in self._satrecs.items():
            pos = propagate_single(sat, timestamp)
            if pos is None:
                continue
            meta = self._metadata.get(norad_id, {})
            results.append({
                "norad_id": norad_id,
                "name": meta.get("name", ""),
                "lat": round(pos["lat"], 4),
                "lon": round(pos["lon"], 4),
                "alt_km": round(pos["alt_km"], 1),
                "shell_km": meta.get("shell_km", 0),
                "status": meta.get("status", "active"),
            })
        return results

    @property
    def count(self) -> int:
        return len(self._satrecs)
