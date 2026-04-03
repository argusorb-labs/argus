"""JPL Horizons Worker — Authoritative ephemeris data for Artemis II.

Polls JPL Horizons API for precise Orion position/velocity vectors.
Used as ground-truth to cross-validate the issinfo.net scraper.

Usage:
    python -m services.telemetry.horizons_worker
"""

from __future__ import annotations

import asyncio
import math
import re
import signal
import sys
import time
from datetime import datetime, timezone

try:
    from services.telemetry.models import TelemetryPoint
except ImportError:
    from models import TelemetryPoint

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
ARTEMIS_II_ID = "-1024"  # SPKID for Artemis II (Orion EM-2)
POLL_INTERVAL = 60  # seconds (Horizons updates are not real-time)

# Earth and Moon radii in km
R_EARTH_KM = 6371.0
R_MOON_KM = 1737.4

# ---------------------------------------------------------------------------
# Horizons API client
# ---------------------------------------------------------------------------


def _get_ssl_context():
    """Get SSL context, falling back to unverified if system certs are broken."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


async def fetch_horizons_vectors(
    start_time: str, stop_time: str, step: str = "5 min"
) -> list[dict]:
    """Fetch state vectors from JPL Horizons.

    Args:
        start_time: ISO format, e.g. "2026-04-03 12:00"
        stop_time: ISO format
        step: Step size, e.g. "5 min", "10 min"

    Returns:
        List of dicts with keys: timestamp, x, y, z, vx, vy, vz (km, km/s)
    """
    import urllib.request
    import urllib.parse
    import json
    import ssl

    params = urllib.parse.urlencode({
        "format": "json",
        "COMMAND": f"'{ARTEMIS_II_ID}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@399'",  # Earth center
        "START_TIME": f"'{start_time}'",
        "STOP_TIME": f"'{stop_time}'",
        "STEP_SIZE": f"'{step}'",
        "VEC_TABLE": "'2'",  # position + velocity
    })

    url = f"{HORIZONS_API}?{params}"

    ctx = _get_ssl_context()
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: urllib.request.urlopen(url, timeout=30, context=ctx).read()
    )

    data = json.loads(response)
    result_text = data.get("result", "")

    return _parse_vectors(result_text)


def _parse_vectors(text: str) -> list[dict]:
    """Parse Horizons vector table output."""
    results = []

    soe = text.find("$$SOE")
    eoe = text.find("$$EOE")
    if soe < 0 or eoe < 0:
        return results

    block = text[soe + 5:eoe].strip()
    lines = block.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for date line: "2461134.000000000 = A.D. 2026-Apr-03 12:00:00.0000 TDB"
        date_match = re.match(
            r"[\d.]+ = A\.D\. (\d{4})-(\w{3})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.\d+ TDB",
            line,
        )
        if date_match and i + 2 < len(lines):
            year, mon_str, day, hour, minute, sec = date_match.groups()

            # Parse month
            months = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
            }
            month = months.get(mon_str, 1)

            dt = datetime(
                int(year), month, int(day),
                int(hour), int(minute), int(sec),
                tzinfo=timezone.utc,
            )
            timestamp = dt.timestamp()

            # Next line: X Y Z
            pos_line = lines[i + 1].strip()
            pos_match = re.findall(r"[+-]?\d+\.\d+E[+-]\d+", pos_line)

            # Next line: VX VY VZ
            vel_line = lines[i + 2].strip()
            vel_match = re.findall(r"[+-]?\d+\.\d+E[+-]\d+", vel_line)

            if len(pos_match) >= 3 and len(vel_match) >= 3:
                x, y, z = float(pos_match[0]), float(pos_match[1]), float(pos_match[2])
                vx, vy, vz = float(vel_match[0]), float(vel_match[1]), float(vel_match[2])

                results.append({
                    "timestamp": timestamp,
                    "x_km": x,
                    "y_km": y,
                    "z_km": z,
                    "vx_kms": vx,
                    "vy_kms": vy,
                    "vz_kms": vz,
                })

            i += 3
        else:
            i += 1

    return results


# ---------------------------------------------------------------------------
# Convert vectors to TelemetryPoint-compatible format
# ---------------------------------------------------------------------------


def vectors_to_telemetry(vec: dict, moon_pos: tuple[float, float, float] | None = None) -> dict:
    """Convert Horizons state vector to telemetry dict.

    Args:
        vec: Dict with x_km, y_km, z_km, vx_kms, vy_kms, vz_kms, timestamp.
        moon_pos: (x, y, z) of Moon in km, if known.

    Returns:
        Dict compatible with TelemetryPoint.to_dict() format.
    """
    x, y, z = vec["x_km"], vec["y_km"], vec["z_km"]
    vx, vy, vz = vec["vx_kms"], vec["vy_kms"], vec["vz_kms"]

    earth_dist = math.sqrt(x * x + y * y + z * z)
    velocity = math.sqrt(vx * vx + vy * vy + vz * vz)

    moon_dist = 0.0
    if moon_pos:
        dx = x - moon_pos[0]
        dy = y - moon_pos[1]
        dz = z - moon_pos[2]
        moon_dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Determine phase from distances
    if earth_dist < 10000:
        phase = "Earth Orbit"
    elif moon_dist > 0 and moon_dist < 50000:
        phase = "Lunar Vicinity"
    elif earth_dist > 200000:
        phase = "Outbound Coast" if velocity > 1.0 else "Return Coast"
    else:
        phase = "Transit"

    # MET placeholder — computed from launch time if known
    met = f"T+{int(vec['timestamp'] - 1775100000)}s"

    return {
        "timestamp": vec["timestamp"],
        "met": met,
        "phase": phase,
        "velocity_kms": velocity,
        "earth_dist_km": earth_dist,
        "moon_dist_km": moon_dist,
        "source": "jpl_horizons",
        # Extra precision fields
        "pos_km": [x, y, z],
        "vel_kms": [vx, vy, vz],
    }


async def fetch_moon_position(timestamp_utc: str) -> tuple[float, float, float] | None:
    """Fetch Moon position from Horizons at a given time."""
    try:
        # Reuse the generic vector fetcher with Moon ID
        import urllib.request
        import urllib.parse
        import json

        params = urllib.parse.urlencode({
            "format": "json",
            "COMMAND": "'301'",
            "OBJ_DATA": "'NO'",
            "MAKE_EPHEM": "'YES'",
            "EPHEM_TYPE": "'VECTORS'",
            "CENTER": "'500@399'",
            "START_TIME": f"'{timestamp_utc}'",
            "STOP_TIME": f"'{timestamp_utc[:10]} 23:59'",
            "STEP_SIZE": "'1440 min'",  # one step = 24h, so only 1 result
            "VEC_TABLE": "'2'",
        })

        url = f"{HORIZONS_API}?{params}"
        ctx = _get_ssl_context()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(url, timeout=30, context=ctx).read()
        )
        data = json.loads(response)
        vecs = _parse_vectors(data.get("result", ""))
        if vecs:
            v = vecs[0]
            return (v["x_km"], v["y_km"], v["z_km"])
    except Exception as e:
        print(f"[HORIZONS] Failed to fetch Moon position: {e}", file=sys.stderr)

    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_horizons_worker(
    on_telemetry=None,
    poll_interval: int = POLL_INTERVAL,
) -> None:
    """Poll Horizons for latest Artemis II state vectors.

    Args:
        on_telemetry: Callback(dict) for each new telemetry point.
        poll_interval: Seconds between API calls.
    """
    print(f"[HORIZONS] Worker starting (interval={poll_interval}s)")
    print(f"[HORIZONS] Target: Artemis II (SPKID {ARTEMIS_II_ID})")

    cycle = 0
    while True:
        cycle += 1
        try:
            now = datetime.now(timezone.utc)
            start = now.strftime("%Y-%m-%d %H:%M")
            # Horizons requires start < stop
            stop = f"{start} + 1 min"

            # Get Moon position first
            moon_pos = await fetch_moon_position(start)

            # Get Orion vectors
            vectors = await fetch_horizons_vectors(start, stop, step="1 min")

            if vectors:
                latest = vectors[-1]
                telem = vectors_to_telemetry(latest, moon_pos)

                print(
                    f"[HORIZONS][{cycle:04d}] "
                    f"v={telem['velocity_kms']:.3f} km/s  "
                    f"Earth={telem['earth_dist_km']:.0f} km  "
                    f"Moon={telem['moon_dist_km']:.0f} km  "
                    f"Phase={telem['phase']}"
                )

                if on_telemetry:
                    on_telemetry(telem)
            else:
                print(f"[HORIZONS][{cycle:04d}] No data returned")

        except Exception as e:
            print(f"[HORIZONS][{cycle:04d}] Error: {e}", file=sys.stderr)

        await asyncio.sleep(poll_interval)


def main() -> None:
    def _shutdown(sig, frame):
        print("\n[HORIZONS] Shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    asyncio.run(run_horizons_worker())


if __name__ == "__main__":
    main()
