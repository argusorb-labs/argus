#!/usr/bin/env python3
"""Parse Space-Track yearly TLE ZIP files into training-ready numpy arrays.

Space-Track provides bulk TLE history as ZIP files (one per year) at:
  https://ln5.sync.com/dl/afd354190/c5cd2q72-a5qjzp4q-nbjdiqkr-cenajuqu

This is the CORRECT way to get bulk historical data (NOT per-object
API queries, which got our account suspended 2026-04-22).

Usage:
    # Parse one year
    python scripts/parse_spacetrack_zip.py --input data/bulk/2025.zip --output data/spacetrack_parsed/

    # Parse all years in a directory
    python scripts/parse_spacetrack_zip.py --input data/bulk/ --output data/spacetrack_parsed/

Output: per-year numpy files + consolidated training splits.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_tle_lines(line1: str, line2: str) -> dict | None:
    """Parse a TLE line pair into orbital elements."""
    try:
        if not line1.startswith("1 ") or not line2.startswith("2 "):
            return None

        norad_id = int(line1[2:7].strip())

        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
        full_year = 1900 + epoch_year if epoch_year >= 57 else 2000 + epoch_year

        inclination = float(line2[8:16].strip())
        eccentricity = float("0." + line2[26:33].strip())
        mean_motion = float(line2[52:63].strip())

        # B* drag term
        bstar_str = line1[53:61].strip()
        if bstar_str and len(bstar_str) >= 2:
            exp_idx = max(bstar_str.rfind('+'), bstar_str.rfind('-'))
            if exp_idx > 0:
                mantissa_str = bstar_str[:exp_idx]
                exp_str = bstar_str[exp_idx:]
                if mantissa_str[0] in '+-':
                    mantissa = float(f"{mantissa_str[0]}0.{mantissa_str[1:]}")
                else:
                    mantissa = float(f"0.{mantissa_str}")
                bstar = mantissa * (10 ** int(exp_str))
            else:
                bstar = float(bstar_str) if bstar_str else 0.0
        else:
            bstar = 0.0

        # Additional orbital elements
        raan = float(line2[17:25].strip())          # Right Ascension of Ascending Node (deg)
        argp = float(line2[34:42].strip())          # Argument of Perigee (deg)
        mean_anomaly = float(line2[43:51].strip())  # Mean Anomaly (deg)

        # n_dot: first derivative of mean motion (rev/day²), in line1[33:43]
        n_dot_str = line1[33:43].strip()
        n_dot = float(n_dot_str) if n_dot_str else 0.0

        # Altitude from mean motion
        import math
        MU = 3.986004418e14
        R_EARTH = 6371.0
        if mean_motion > 0:
            n = mean_motion * 2 * math.pi / 86400
            a = (MU / (n * n)) ** (1 / 3)
            alt_km = a / 1000 - R_EARTH
        else:
            alt_km = 0

        return {
            "norad_id": norad_id,
            "epoch_year": full_year,
            "epoch_day": epoch_day,
            "mean_motion": mean_motion,
            "eccentricity": eccentricity,
            "inclination": inclination,
            "bstar": bstar,
            "alt_km": alt_km,
            "raan": raan,
            "argp": argp,
            "mean_anomaly": mean_anomaly,
            "n_dot": n_dot,
        }
    except (ValueError, IndexError):
        return None


def parse_zip_file(zip_path: Path) -> dict[int, list[dict]]:
    """Parse a Space-Track yearly ZIP file.

    Returns: {norad_id: [tle_records sorted by epoch]}
    """
    satellites: dict[int, list[dict]] = defaultdict(list)
    total_parsed = 0
    total_errors = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if not name.endswith('.tle') and not name.endswith('.txt'):
                continue

            with zf.open(name) as f:
                text = f.read().decode('utf-8', errors='ignore')

            lines = [l.strip() for l in text.split('\n') if l.strip()]

            i = 0
            while i + 1 < len(lines):
                line1 = lines[i]
                line2 = lines[i + 1]

                # Skip name lines
                if not line1.startswith("1 "):
                    i += 1
                    continue

                if not line2.startswith("2 "):
                    i += 1
                    continue

                parsed = parse_tle_lines(line1, line2)
                if parsed:
                    satellites[parsed["norad_id"]].append(parsed)
                    total_parsed += 1
                else:
                    total_errors += 1

                i += 2

    # Sort each satellite's records by epoch
    for nid in satellites:
        satellites[nid].sort(key=lambda r: (r["epoch_year"], r["epoch_day"]))

    print(f"  parsed {total_parsed:,} TLEs from {len(satellites):,} objects "
          f"({total_errors} errors)")
    return dict(satellites)


def satellites_to_sequences(
    satellites: dict[int, list[dict]],
    seq_len: int = 100,
    stride: int = 50,
) -> tuple[np.ndarray, list[int]]:
    """Convert per-satellite TLE records into fixed-length sequences.

    Returns (X, norad_ids) where X shape is (N, seq_len, 11).
    Features: [epoch_h, mean_motion, eccentricity, inclination, bstar, alt_km,
               dt_hours, raan, argp, mean_anomaly, n_dot]
    """
    all_sequences = []
    all_norad_ids = []

    for norad_id, records in satellites.items():
        if len(records) < seq_len:
            continue

        def _epoch_hours(r):
            return (r["epoch_year"] * 365.25 + r["epoch_day"]) * 24.0

        n = len(records)
        arr = np.zeros((n, 11))
        epoch_h = np.array([_epoch_hours(r) for r in records])

        for j, r in enumerate(records):
            arr[j, 1] = r["mean_motion"]
            arr[j, 2] = r["eccentricity"]
            arr[j, 3] = r["inclination"]
            arr[j, 4] = r["bstar"]
            arr[j, 5] = r["alt_km"]
            arr[j, 7] = r.get("raan", 0.0)
            arr[j, 8] = r.get("argp", 0.0)
            arr[j, 9] = r.get("mean_anomaly", 0.0)
            arr[j, 10] = r.get("n_dot", 0.0)

        arr[:, 0] = epoch_h

        # dt_hours: time since previous TLE (0 for first)
        arr[0, 6] = 0.0
        arr[1:, 6] = np.diff(epoch_h)
        arr[:, 6] = np.clip(arr[:, 6], 0.0, 240.0)

        # Sliding windows
        for start in range(0, n - seq_len + 1, stride):
            window = arr[start:start + seq_len].copy()
            window[:, 0] -= window[0, 0]  # relative epoch within window
            # Recompute dt_hours for first element of window
            window[0, 6] = 0.0
            all_sequences.append(window)
            all_norad_ids.append(norad_id)

    if not all_sequences:
        return np.array([]), []

    return np.array(all_sequences), all_norad_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse Space-Track yearly ZIP files into numpy training data.",
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Single ZIP file or directory of ZIPs")
    parser.add_argument("--output", type=Path, default=Path("data/spacetrack_parsed"),
                        help="Output directory for numpy files")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--years", help="Comma-separated years to process (e.g. 2024,2025)")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    # Find ZIP files
    if args.input.is_file():
        zip_files = [args.input]
    else:
        zip_files = sorted(args.input.glob("*.zip"))

    if args.years:
        year_filter = set(args.years.split(","))
        zip_files = [z for z in zip_files if any(y in z.name for y in year_filter)]

    if not zip_files:
        print(f"No ZIP files found in {args.input}")
        return 1

    print(f"Found {len(zip_files)} ZIP files to process")

    all_X = []
    total_tles = 0

    for zf in zip_files:
        print(f"\n[{zf.name}] ({zf.stat().st_size / 1e6:.0f} MB)")
        t0 = time.time()

        satellites = parse_zip_file(zf)
        total_tles += sum(len(v) for v in satellites.values())

        X, norad_ids = satellites_to_sequences(
            satellites, seq_len=args.seq_len, stride=args.stride
        )

        if len(X) > 0:
            # Save per-year
            year_name = zf.stem
            np.save(args.output / f"X_{year_name}.npy", X)
            all_X.append(X)
            print(f"  → {len(X):,} sequences, {X.nbytes / 1e6:.0f} MB, "
                  f"{time.time() - t0:.1f}s")
        else:
            print(f"  → no sequences (too few TLEs per satellite)")

    # Consolidate all years
    if all_X:
        X_all = np.concatenate(all_X)
        np.save(args.output / "X_all.npy", X_all)

        print(f"\n{'=' * 50}")
        print(f"Total: {total_tles:,} TLEs → {len(X_all):,} sequences")
        print(f"Shape: {X_all.shape}")
        print(f"Size: {X_all.nbytes / 1e9:.2f} GB")
        print(f"Saved to {args.output}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
