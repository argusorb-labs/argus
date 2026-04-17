#!/usr/bin/env python3
"""Pull TLE history from Space-Track for a specific NORAD ID.

Usage:
    export SPACETRACK_USER='your@email.com'
    export SPACETRACK_PASS='yourpassword'
    python scripts/spacetrack_history.py 64157

Outputs a TSV time series of orbital elements to stdout.
"""

from __future__ import annotations

import os
import sys

import httpx

BASE = "https://www.space-track.org"
LOGIN_URL = f"{BASE}/ajaxauth/login"


def main(norad_id: int) -> int:
    user = os.environ.get("SPACETRACK_USER")
    pw = os.environ.get("SPACETRACK_PASS")
    if not user or not pw:
        print("Set SPACETRACK_USER and SPACETRACK_PASS env vars", file=sys.stderr)
        return 1

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        # Authenticate
        resp = client.post(LOGIN_URL, data={
            "identity": user,
            "password": pw,
        })
        if resp.status_code != 200:
            print(f"Login failed: {resp.status_code}", file=sys.stderr)
            return 1

        # Pull GP history as JSON (easier to parse than TLE for analysis)
        query_url = (
            f"{BASE}/basicspacedata/query/class/gp_history"
            f"/NORAD_CAT_ID/{norad_id}"
            f"/orderby/EPOCH asc"
            f"/format/json"
        )
        resp = client.get(query_url)
        if resp.status_code != 200:
            print(f"Query failed: {resp.status_code} {resp.text[:200]}", file=sys.stderr)
            return 1

        records = resp.json()
        if not records:
            print(f"No records for NORAD {norad_id}", file=sys.stderr)
            return 1

        # TSV header
        print("\t".join([
            "epoch", "epoch_jd", "mean_motion", "eccentricity",
            "inclination", "bstar", "semi_major_axis_km",
            "apoapsis_km", "periapsis_km", "period_min",
        ]))

        for r in records:
            print("\t".join([
                r.get("EPOCH", ""),
                str(r.get("EPOCH_JD", "")),
                str(r.get("MEAN_MOTION", "")),
                str(r.get("ECCENTRICITY", "")),
                str(r.get("INCLINATION", "")),
                str(r.get("BSTAR", "")),
                str(r.get("SEMIMAJOR_AXIS", "")),
                str(r.get("APOAPSIS", "")),
                str(r.get("PERIAPSIS", "")),
                str(r.get("PERIOD", "")),
            ]))

        print(f"\n# {len(records)} records for NORAD {norad_id}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} NORAD_ID", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(int(sys.argv[1])))
