"""Supplemental GP Fetcher — pulls operator-sourced TLEs from Celestrak.

Runs alongside the primary TLE fetcher but ingests supplemental GP data
(TLEs derived from operator-provided precise tracking, not NORAD radar).
Currently fetches Planet Labs data; more sources can be added later.

Why: supplemental GP elements are generated from GPS-quality tracking
(~10-50 m) rather than radar (~1 km). Comparing standard NORAD TLEs
against these "better TLEs" for the same satellite reveals SGP4's
systematic errors — training data for the precision calibration model
(see argus-internal/design/PRECISION_CALIBRATION.md).
"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx

try:
    from services.telemetry.store import StarlinkStore
    from services.telemetry.tle_fetcher import parse_tle_text
except ImportError:
    from store import StarlinkStore  # type: ignore
    from tle_fetcher import parse_tle_text  # type: ignore

# Each source: (name, Celestrak supplemental URL)
SUPGP_SOURCES = [
    (
        "planet",
        "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=planet&FORMAT=tle",
    ),
    (
        "oneweb",
        "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=oneweb&FORMAT=tle",
    ),
    (
        "spire",
        "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=spire&FORMAT=tle",
    ),
    (
        "iridium",
        "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=iridium-NEXT&FORMAT=tle",
    ),
]

FETCH_INTERVAL = 8 * 3600  # same cadence as Starlink fetcher


async def _fetch_url(url: str) -> str:
    async with httpx.AsyncClient(timeout=60, verify=True) as client:
        resp = await client.get(url, headers={
            "User-Agent": "argusorb/0.2 (supgp-fetcher)",
        })
        resp.raise_for_status()
        return resp.text


async def run_supgp_fetcher(
    store: StarlinkStore,
    interval: int = FETCH_INTERVAL,
) -> None:
    """Fetch supplemental GP TLEs periodically and store."""
    sources_str = ", ".join(s[0] for s in SUPGP_SOURCES)
    print(f"[SUPGP] Fetcher starting (sources={sources_str}, interval={interval}s)")

    cycle = 0
    while True:
        cycle += 1

        for source_name, url in SUPGP_SOURCES:
            t0 = time.perf_counter()
            try:
                text = await _fetch_url(url)
                tles, parse_errors = parse_tle_text(text)
                new_count = store.upsert_supgp_tles(tles, source=source_name)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                print(
                    f"[SUPGP][{cycle:04d}] {source_name}: {len(tles)} parsed "
                    f"({new_count} new, {parse_errors} errors) in {elapsed_ms}ms"
                )
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                print(
                    f"[SUPGP][{cycle:04d}] {source_name}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

        if cycle == 1:
            await asyncio.sleep(10)
        else:
            await asyncio.sleep(interval)
