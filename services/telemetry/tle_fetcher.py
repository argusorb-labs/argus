"""TLE Fetcher — pulls Starlink TLEs from Celestrak.

Runs every 8 hours. Parses TLE triplets, extracts orbital parameters,
classifies orbital shell, stores in SQLite, archives raw response to disk,
and records every attempt (success or failure) in fetch_log.
"""

from __future__ import annotations

import asyncio
import gzip
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    from services.telemetry.store import StarlinkStore
    from services.telemetry.tle_validator import (
        validate_tle_structure,
        validate_tle_physics,
    )
    from services.brain.orbital_analyzer import (
        analyze_constellation,
        detect_tle_gaps,
    )
except ImportError:
    from store import StarlinkStore  # type: ignore
    from tle_validator import (  # type: ignore
        validate_tle_structure,
        validate_tle_physics,
    )
    from brain.orbital_analyzer import (  # type: ignore
        analyze_constellation,
        detect_tle_gaps,
    )

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=tle"
FETCH_INTERVAL = 8 * 3600  # 8 hours
RAW_DIR = Path(os.environ.get("ARGUS_RAW_DIR", "data/raw"))

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


def _parse_tle_float(s: str) -> float:
    """Parse TLE's compact float format used by B* and 2nd derivative fields.

    The format packs a signed mantissa with an implied leading decimal point
    plus a signed single-digit exponent into 8 characters. Examples from the
    wild:

        ' 14452-3' -> 0.14452 × 10^-3 =  1.4452e-4
        '-27482-3' -> -0.27482 × 10^-3 = -2.7482e-4
        ' 00000+0' -> 0.0
        '+12345-5' -> 0.12345 × 10^-5 =  1.2345e-6

    The last two characters are always [+-][0-9] (exponent sign + digit).
    The remaining prefix is the mantissa, with an implied "0." inserted
    after its leading sign (or at the front if unsigned).

    Falls back to plain float() when the last two chars are not a signed
    digit, so the same helper works for TLE fields that use a normal
    decimal representation (like the 1st derivative of mean motion).
    """
    s = s.strip()
    if not s:
        return 0.0

    if len(s) >= 2 and s[-2] in "+-" and s[-1].isdigit():
        exp = int(s[-2:])
        mantissa_str = s[:-2]
    else:
        return float(s)

    if not mantissa_str:
        return 0.0
    if mantissa_str[0] in "+-":
        sign = mantissa_str[0]
        digits = mantissa_str[1:]
        mantissa = float(f"{sign}0.{digits}") if digits else 0.0
    else:
        mantissa = float(f"0.{mantissa_str}")
    return mantissa * (10 ** exp)


def parse_tle_text(text: str) -> tuple[list[dict], int]:
    """Parse Celestrak TLE text (name, line1, line2 triplets) with cleaning.

    Every triplet is run through validate_tle_structure (checksum, length,
    NORAD match) and validate_tle_physics (range bounds on the parsed
    elements). Rejections are counted by reason and the aggregated breakdown
    is printed to stderr — stored in the fetch_log only as a total count.

    Returns (parsed_tles, error_count). Rejections and parse exceptions both
    count toward error_count. The raw bytes are archived by the caller
    regardless, so a rejection here is recoverable via replay.
    """
    lines = [l.strip().replace("\r", "") for l in text.strip().split("\n") if l.strip()]
    results: list[dict] = []
    errors = 0
    rejections: dict[str, int] = {}

    def _reject(reason: str) -> None:
        nonlocal errors
        errors += 1
        rejections[reason] = rejections.get(reason, 0) + 1

    i = 0
    while i + 2 < len(lines):
        name_line = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        # Prefix check is inline (not in validator) so we can advance one line
        # instead of three when lines are misaligned — this lets the parser
        # recover from junk blocks between valid triplets without losing the
        # next valid triplet.
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            _reject("prefix")
            i += 1
            continue

        ok, reason = validate_tle_structure(line1, line2)
        if not ok:
            _reject(reason or "structure")
            i += 3
            continue

        try:
            norad_id = int(line1[2:7].strip())
            intl_des = line1[9:17].strip()

            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            full_year = 1900 + epoch_year if epoch_year >= 57 else 2000 + epoch_year
            from sgp4.api import jday
            jan1_jd = jday(full_year, 1, 1, 0, 0, 0)[0]
            epoch_jd = jan1_jd + epoch_day - 1

            inclination = float(line2[8:16].strip())
            eccentricity = float("0." + line2[26:33].strip())
            mean_motion = float(line2[52:63].strip())

            # B* drag term: TLE line 1 columns 54-61 (0-indexed 53:61).
            # This is the only field that lets downstream classifiers
            # distinguish "atmospheric anomaly" from "commanded maneuver" —
            # see argus-internal/design/STRATEGY.md. Persisted unvalidated
            # for now; range bounds will be set after we have a production
            # distribution to calibrate against.
            bstar = _parse_tle_float(line1[53:61])

            alt_km = mean_motion_to_alt_km(mean_motion)
            shell_km = classify_shell(alt_km)
            launch_group = intl_des[:8] if intl_des else ""

            parsed = {
                "norad_id": norad_id,
                "name": name_line.strip(),
                "line1": line1,
                "line2": line2,
                "epoch_jd": epoch_jd,
                "inclination": inclination,
                "mean_motion": mean_motion,
                "eccentricity": eccentricity,
                "bstar": bstar,
                "intl_designator": intl_des,
                "shell_km": shell_km,
                "launch_group": launch_group,
                "alt_km": alt_km,
            }
        except (ValueError, IndexError):
            _reject("parse_exception")
            i += 3
            continue

        ok, reason = validate_tle_physics(parsed)
        if not ok:
            _reject(reason or "physics")
            i += 3
            continue

        results.append(parsed)
        i += 3

    if rejections:
        print(
            f"[TLE PARSE] rejected {sum(rejections.values())}: {dict(sorted(rejections.items()))}",
            file=sys.stderr,
        )

    return results, errors


def archive_raw(text: str, raw_dir: Path = RAW_DIR) -> Path:
    """Write raw Celestrak response to data/raw/YYYYMMDD/celestrak-TS.tle.gz.

    This is our 'never lose an update' insurance — if parsing has bugs or
    Celestrak changes format, we can replay history from these files.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    day_dir = raw_dir / ts[:8]
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"celestrak-{ts}.tle.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)
    return path


async def fetch_celestrak() -> str:
    """Fetch TLE text from Celestrak."""
    async with httpx.AsyncClient(timeout=30, verify=True) as client:
        resp = await client.get(CELESTRAK_URL, headers={
            "User-Agent": "argusorb/0.2 (starlink-tracker)",
        })
        resp.raise_for_status()
        return resp.text


async def run_tle_fetcher(
    store: StarlinkStore,
    on_complete=None,
    interval: int = FETCH_INTERVAL,
) -> None:
    """Fetch TLEs periodically, archive raw, store parsed, log the attempt."""
    print(f"[TLE] Fetcher starting (interval={interval}s, raw_dir={RAW_DIR})")

    cycle = 0
    while True:
        cycle += 1
        cycle_start_ts = time.time()
        t_start = time.perf_counter()
        status = "ok"
        error_msg: str | None = None
        archive_path: str | None = None
        text = ""
        tles: list[dict] = []
        parse_errors = 0
        new_count = 0
        new_labels = 0

        try:
            text = await fetch_celestrak()
            archive_path = str(archive_raw(text))
            tles, parse_errors = parse_tle_text(text)
            new_count = store.upsert_tles(tles)

            # Label any satellite that got a fresh TLE this cycle.
            # rule_v1 is idempotent (unique index on anomaly), so replays
            # are free — new_labels counts only the newly-written rows.
            try:
                labels = analyze_constellation(store, since_ts=cycle_start_ts)
                new_labels = len(labels)
            except Exception as e:
                print(
                    f"[TLE][{cycle:04d}] rule_v1 pass failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

            # Check for satellites that have gone silent (>24h without TLE).
            try:
                gaps = detect_tle_gaps(store)
                if gaps:
                    print(
                        f"[TLE][{cycle:04d}] ⚠ {len(gaps)} satellites silent >24h: "
                        + ", ".join(
                            f"{g.get('name') or g['norad_id']} ({g['gap_hours']:.0f}h)"
                            for g in gaps[:5]
                        )
                        + (" ..." if len(gaps) > 5 else "")
                    )
            except Exception as e:
                print(
                    f"[TLE][{cycle:04d}] gap detection failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

            elapsed_ms = int((time.perf_counter() - t_start) * 1000)
            print(
                f"[TLE][{cycle:04d}] ok: {len(tles)} parsed "
                f"({new_count} new, {parse_errors} errors, "
                f"{new_labels} labels) in {elapsed_ms}ms"
            )

            if on_complete:
                on_complete(len(tles), new_count)

        except Exception as e:
            status = "error"
            error_msg = f"{type(e).__name__}: {e}"
            elapsed_ms = int((time.perf_counter() - t_start) * 1000)
            print(f"[TLE][{cycle:04d}] {error_msg}", file=sys.stderr)

        try:
            store.log_fetch(
                status=status,
                http_bytes=len(text.encode("utf-8")) if text else 0,
                parsed_count=len(tles),
                new_tle_count=new_count,
                parse_errors=parse_errors,
                duration_ms=elapsed_ms,
                error_msg=error_msg,
                raw_archive_path=archive_path,
            )
        except Exception as e:
            print(f"[TLE][{cycle:04d}] fetch_log write failed: {e}", file=sys.stderr)

        if cycle == 1:
            await asyncio.sleep(10)
        else:
            await asyncio.sleep(interval)
