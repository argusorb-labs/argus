"""TLE validation — checksum, structural sanity, physical range bounds.

This is the cleaning layer between parse_tle_text and the store. Every TLE
that passes both validate_tle_structure and validate_tle_physics is
considered "clean enough to persist." Everything else is rejected with a
categorized reason string that the caller counts and logs.

Design notes:
- Raw bytes are NEVER discarded. The raw Celestrak response is archived to
  data/raw/YYYYMMDD/*.tle.gz regardless of validation outcome, so even TLEs
  rejected here can be replayed later if a parser bug is discovered.
- Physical bounds are set wide enough to not reject real (but unusual)
  satellites — only things that are *obviously* wrong (ecc > 1, altitude
  inside the Earth, etc.) fall out. Suspicious-but-possible cases are not
  our job here; those get flagged by rule_v1 in the anomaly table instead.
- Rejection reasons are short stable tags. Aggregated counts go into
  fetch logs and can be graphed over time to detect ingestion regressions.
"""

from __future__ import annotations

# ── Physical bounds ──
# Wide enough to never reject a real satellite; tight enough to catch
# parser/catalog bugs with high signal-to-noise.

MIN_MEAN_MOTION = 0.5     # rev/day — below this is barely sub-GEO; real LEO is ~11-18
MAX_MEAN_MOTION = 20.0    # rev/day — above this the orbit is inside the Earth

MIN_ECCENTRICITY = 0.0    # inclusive
MAX_ECCENTRICITY = 1.0    # exclusive — ecc == 1 is parabolic, > 1 hyperbolic (escape)

MIN_INCLINATION = 0.0     # inclusive
MAX_INCLINATION = 180.0   # inclusive — 180° is a fully retrograde polar orbit

MIN_ALTITUDE_KM = 150.0   # below this, atmospheric drag gives days of orbital life at most
MAX_ALTITUDE_KM = 50000.0 # well beyond GEO (~36,000 km); anything higher is a parse bug


def _compute_checksum(line: str) -> int:
    """Standard TLE checksum: sum of digits in cols 1-68, plus 1 per '-', mod 10.

    Reference: https://celestrak.org/columns/v04n03/ — "Checksum" section.
    """
    s = 0
    for c in line[:68]:
        if c.isdigit():
            s += int(c)
        elif c == "-":
            s += 1
    return s % 10


def validate_tle_structure(line1: str, line2: str) -> tuple[bool, str | None]:
    """Check length, checksum, and NORAD ID match between the two lines.

    Line prefixes ('1 ' / '2 ') are assumed to be already validated upstream
    — callers do the prefix check first so they can advance one line on
    prefix mismatch (to resync across misaligned triplets) instead of
    discarding a three-line block.

    Returns:
        (True, None) if the pair is structurally sound, otherwise
        (False, short_reason) where short_reason is a stable tag used for
        rejection aggregation.
    """
    if len(line1) != 69 or len(line2) != 69:
        return False, "length"

    try:
        stored1 = int(line1[68])
        stored2 = int(line2[68])
    except (ValueError, IndexError):
        return False, "checksum_format"
    if _compute_checksum(line1) != stored1:
        return False, "checksum_l1"
    if _compute_checksum(line2) != stored2:
        return False, "checksum_l2"

    try:
        nid1 = int(line1[2:7])
        nid2 = int(line2[2:7])
    except ValueError:
        return False, "norad_format"
    if nid1 != nid2:
        return False, "norad_mismatch"

    return True, None


def validate_tle_physics(parsed: dict) -> tuple[bool, str | None]:
    """Range checks on the parsed element dict.

    Called after a successful parse, before the tle is appended to the
    result list. A failure here almost always means a parser bug or a bad
    catalog entry, not a real satellite doing something strange.
    """
    mm = parsed.get("mean_motion", 0)
    if not (MIN_MEAN_MOTION < mm < MAX_MEAN_MOTION):
        return False, "mean_motion_range"

    ecc = parsed.get("eccentricity", -1)
    if not (MIN_ECCENTRICITY <= ecc < MAX_ECCENTRICITY):
        return False, "eccentricity_range"

    incl = parsed.get("inclination", -1)
    if not (MIN_INCLINATION <= incl <= MAX_INCLINATION):
        return False, "inclination_range"

    alt = parsed.get("alt_km", -1)
    if not (MIN_ALTITUDE_KM <= alt < MAX_ALTITUDE_KM):
        return False, "altitude_range"

    return True, None
