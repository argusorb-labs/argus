"""TLE Fetcher — pulls Starlink TLEs from Celestrak.

Runs every 8 hours. Parses TLE triplets, extracts orbital parameters,
classifies orbital shell, stores in SQLite.
"""

from __future__ import annotations

import asyncio
import math
import re
import signal
import sys
import time

import httpx

try:
    from services.telemetry.store import StarlinkStore
except ImportError:
    from store import StarlinkStore

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=tle"
FETCH_INTERVAL = 8 * 3600  # 8 hours

# Earth constants for shell classification
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH_KM = 6371.0


def mean_motion_to_alt_km(mean_motion_rev_day: float) -> float:
    """Convert mean motion (rev/day) to altitude above Earth (km)."""
    if mean_motion_rev_day <= 0:
        return 0
    n = mean_motion_rev_day * 2 * math.pi / 86400  # rad/s
    a = (MU_EARTH / (n * n)) ** (1 / 3)  # semi-major axis in meters
    return a / 1000 - R_EARTH_KM


def classify_shell(alt_km: float) -> float:
    """Classify into Starlink orbital shell."""
    if alt_km < 300:
        return 0  # deorbiting/decayed
    if 330 <= alt_km <= 360:
        return 340
    if 520 <= alt_km <= 545:
        return 530
    if 545 <= alt_km <= 560:
        return 550
    if 560 <= alt_km <= 580:
        return 570
    return round(alt_km / 10) * 10  # round to nearest 10


def parse_tle_text(text: str) -> list[dict]:
    """Parse Celestrak TLE text (name, line1, line2 triplets)."""
    lines = [l.strip().replace("\r", "") for l in text.strip().split("\n") if l.strip()]
    results = []

    i = 0
    while i + 2 < len(lines):
        name_line = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        # Validate TLE format
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            i += 1
            continue

        try:
            norad_id = int(line1[2:7].strip())
            intl_des = line1[9:17].strip()

            # Epoch: year (2-digit) + day of year (fractional)
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            if epoch_year >= 57:
                full_year = 1900 + epoch_year
            else:
                full_year = 2000 + epoch_year
            # Convert to Julian Date
            from sgp4.api import jday
            jan1_jd = jday(full_year, 1, 1, 0, 0, 0)[0]
            epoch_jd = jan1_jd + epoch_day - 1

            # Orbital elements from line 2
            inclination = float(line2[8:16].strip())
            eccentricity = float("0." + line2[26:33].strip())
            mean_motion = float(line2[52:63].strip())

            alt_km = mean_motion_to_alt_km(mean_motion)
            shell_km = classify_shell(alt_km)

            # Launch group from international designator
            launch_group = intl_des[:8] if intl_des else ""

            results.append({
                "norad_id": norad_id,
                "name": name_line.strip(),
                "line1": line1,
                "line2": line2,
                "epoch_jd": epoch_jd,
                "inclination": inclination,
                "mean_motion": mean_motion,
                "eccentricity": eccentricity,
                "intl_designator": intl_des,
                "shell_km": shell_km,
                "launch_group": launch_group,
                "alt_km": alt_km,
            })
        except (ValueError, IndexError) as e:
            pass  # skip malformed TLEs

        i += 3

    return results


async def fetch_celestrak() -> str:
    """Fetch TLE text from Celestrak."""
    async with httpx.AsyncClient(timeout=30, verify=True) as client:
        resp = await client.get(CELESTRAK_URL, headers={
            "User-Agent": "selene-insight/0.2 (starlink-tracker)",
        })
        resp.raise_for_status()
        return resp.text


async def run_tle_fetcher(
    store: StarlinkStore,
    on_complete=None,
    interval: int = FETCH_INTERVAL,
) -> None:
    """Fetch TLEs periodically and store in SQLite."""
    print(f"[TLE] Fetcher starting (interval={interval}s)")

    cycle = 0
    while True:
        cycle += 1
        try:
            t0 = time.perf_counter()
            text = await fetch_celestrak()
            tles = parse_tle_text(text)
            elapsed_fetch = time.perf_counter() - t0

            t1 = time.perf_counter()
            new_count = store.upsert_tles(tles)
            elapsed_store = time.perf_counter() - t1

            print(
                f"[TLE][{cycle:04d}] Fetched {len(tles)} satellites "
                f"({new_count} new TLEs) in {elapsed_fetch:.1f}s fetch + "
                f"{elapsed_store:.1f}s store"
            )

            if on_complete:
                on_complete(len(tles), new_count)

        except Exception as e:
            print(f"[TLE][{cycle:04d}] Error: {e}", file=sys.stderr)

        if cycle == 1:
            # First fetch: short wait then continue
            await asyncio.sleep(10)
        else:
            await asyncio.sleep(interval)
