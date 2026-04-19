"""Space-Track authenticated fetcher for NORAD catalog GP elements.

Celestrak's gp.php endpoint added WAF filtering that blocks server
User-Agents; we bypass it by pulling from the upstream (Space-Track)
directly. Supplemental GP (operator-provided) stays on Celestrak.

Credentials via SPACETRACK_USER / SPACETRACK_PASS env vars.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time

import httpx

try:
    from services.telemetry.store import StarlinkStore
    from services.telemetry.tle_fetcher import parse_tle_text
except ImportError:
    from store import StarlinkStore  # type: ignore
    from tle_fetcher import parse_tle_text  # type: ignore

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
LOGOUT_URL = "https://www.space-track.org/ajaxauth/logout"
STARLINK_QUERY = (
    "https://www.space-track.org/basicspacedata/query/class/gp"
    "/OBJECT_NAME/~~STARLINK"
    "/orderby/NORAD_CAT_ID%20asc/format/3le"
)

FETCH_INTERVAL = 8 * 3600
SOURCE_NAME = "norad_catalog_starlink"
USER_AGENT = "argusorb/0.2 (spacetrack-fetcher)"

# Space-Track 3LE name line is "0 NAME"; strip prefix to match parse_tle_text's expected format.
_NAME_PREFIX_RE = re.compile(r"^0 ", flags=re.MULTILINE)


async def _fetch_once(user: str, password: str) -> str:
    async with httpx.AsyncClient(
        timeout=120, verify=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        r = await client.post(
            LOGIN_URL, data={"identity": user, "password": password}
        )
        r.raise_for_status()
        r = await client.get(STARLINK_QUERY)
        r.raise_for_status()
        text = _NAME_PREFIX_RE.sub("", r.text)
        try:
            await client.get(LOGOUT_URL)
        except Exception:
            pass
        return text


async def run_spacetrack_fetcher(
    store: StarlinkStore,
    interval: int = FETCH_INTERVAL,
) -> None:
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")
    if not user or not password:
        print(
            "[SPACETRACK] SPACETRACK_USER/SPACETRACK_PASS not set — fetcher disabled",
            file=sys.stderr,
        )
        return

    print(f"[SPACETRACK] Fetcher starting (user={user}, interval={interval}s)")
    cycle = 0
    while True:
        cycle += 1
        t0 = time.perf_counter()
        try:
            text = await _fetch_once(user, password)
            tles, parse_errors = parse_tle_text(text)
            new_count = store.upsert_supgp_tles(tles, source=SOURCE_NAME)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            print(
                f"[SPACETRACK][{cycle:04d}] {len(tles)} parsed "
                f"({new_count} new, {parse_errors} errors) in {elapsed_ms}ms"
            )
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            print(
                f"[SPACETRACK][{cycle:04d}] {type(e).__name__}: {e} ({elapsed_ms}ms)",
                file=sys.stderr,
            )
        await asyncio.sleep(interval)
