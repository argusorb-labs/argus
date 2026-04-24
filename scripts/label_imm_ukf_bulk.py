#!/usr/bin/env python3
"""Label Space-Track TLE data with IMM-UKF for high-quality training labels.

Strategy: only label satellites that have enough TLE history for IMM-UKF.
This gives us precision labels on real orbital dynamics — the moat data.

Approach:
  1. Re-parse original TLE ZIPs (preserving line1/line2 for SGP4)
  2. Group by NORAD ID, sort by epoch
  3. Run IMM-UKF on each satellite's history
  4. Also compute rule_v1 label for comparison
  5. Output: multi-label dataset (rule_v1 + imm_ukf per timestep)

Usage:
    python scripts/label_imm_ukf_bulk.py \
        --zip-input ~/Downloads/TLEs.zip \
        --output data/imm_ukf_labeled/ \
        --workers 8

    # Start with recent data (fastest, most relevant)
    python scripts/label_imm_ukf_bulk.py \
        --zip-input ~/Downloads/TLEs.zip \
        --output data/imm_ukf_labeled/ \
        --years 2024,2025 \
        --max-satellites 1000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.label_spacetrack_bulk import label_pair
from scripts.parse_spacetrack_zip import parse_tle_lines


# ── IMM-UKF classification without database dependency ──

def classify_from_tles(
    tle_records: list[dict],
    norad_id: int,
) -> list[dict]:
    """Run IMM-UKF over a satellite's TLE history (no database needed).

    Args:
        tle_records: list of dicts with keys:
            line1, line2, epoch_year, epoch_day, mean_motion,
            eccentricity, inclination, bstar, alt_km
        norad_id: NORAD catalog ID

    Returns:
        list of dicts per timestep with rule_v1 and imm_ukf labels
    """
    from services.brain.dynamics import tle_to_state, R_EARTH
    from services.brain.imm_classifier import (
        create_imm, _batch_fx_wrapper, MODEL_NAMES,
    )

    if len(tle_records) < 2:
        return []

    # Initialize from first TLE
    first = tle_records[0]
    state0 = tle_to_state(first["line1"], first["line2"])
    if state0 is None:
        return []

    alt0_km = np.linalg.norm(state0[:3]) / 1000 - R_EARTH / 1000

    # Skip very low orbits (atmosphere model too crude)
    if alt0_km < 300.0:
        return []

    imm = create_imm(state0, alt_km=alt0_km)
    results = []

    # First record has no predecessor for rule_v1
    results.append({
        "norad_id": norad_id,
        "epoch_year": first["epoch_year"],
        "epoch_day": first["epoch_day"],
        "alt_km": first["alt_km"],
        "rule_v1": 0,
        "imm_ukf": 0,  # 0=station_keeping
        "imm_probs": [1.0, 0.0, 0.0],
        "imm_confidence": 1.0,
    })

    for i in range(1, len(tle_records)):
        prev = tle_records[i - 1]
        curr = tle_records[i]

        # rule_v1 label
        old_feat = np.array([0.0, prev["mean_motion"], prev["eccentricity"],
                             prev["inclination"], prev["bstar"], prev["alt_km"]])
        new_feat = np.array([0.0, curr["mean_motion"], curr["eccentricity"],
                             curr["inclination"], curr["bstar"], curr["alt_km"]])
        rv1_label = label_pair(old_feat, new_feat)

        # Time delta
        dt_days = ((curr["epoch_year"] - prev["epoch_year"]) * 365.25
                   + curr["epoch_day"] - prev["epoch_day"])
        dt_seconds = dt_days * 86400.0

        imm_label = 0
        imm_probs = [1.0, 0.0, 0.0]
        imm_conf = 1.0

        if 0 < dt_seconds < 7 * 86400:
            bstar = curr.get("bstar", 0.0) or 0.0
            fx_args = [(bstar,)] * 3

            try:
                imm.predict(
                    dt=dt_seconds, fx_args_per_model=fx_args,
                    batch_fx=_batch_fx_wrapper,
                )
                obs_state = tle_to_state(curr["line1"], curr["line2"])
                if obs_state is not None:
                    imm.update(obs_state)

                    probs = imm.model_probabilities
                    best = imm.most_likely_model
                    imm_probs = [float(probs[j]) for j in range(3)]
                    imm_conf = float(probs[best])

                    # Map IMM model to label: 0=normal, 1=maneuver, 2=decay
                    if best == 1 and probs[1] > 0.3:
                        imm_label = 1  # maneuver
                    elif best == 2 and probs[2] > 0.3:
                        imm_label = 2  # decay
                    else:
                        imm_label = 0  # normal/station-keeping
            except Exception:
                pass  # Keep defaults on failure

        results.append({
            "norad_id": norad_id,
            "epoch_year": curr["epoch_year"],
            "epoch_day": curr["epoch_day"],
            "alt_km": curr["alt_km"],
            "rule_v1": rv1_label,
            "imm_ukf": imm_label,
            "imm_probs": imm_probs,
            "imm_confidence": imm_conf,
        })

    return results


def _worker(args):
    """Worker for multiprocessing."""
    norad_id, tle_records = args
    try:
        return norad_id, classify_from_tles(tle_records, norad_id)
    except Exception as e:
        return norad_id, []


def _scan_norad_counts(txt_dir: Path, year_filter: set[str] | None = None
                       ) -> tuple[dict[int, int], list[Path]]:
    """Pass 1: count TLEs per NORAD ID without storing data (low memory)."""
    from collections import Counter
    counts: Counter = Counter()
    files = sorted(txt_dir.glob("*.txt"))
    if year_filter:
        files = [f for f in files if any(y in f.name for y in year_filter)]

    for fpath in files:
        print(f"  Scanning {fpath.name}...")
        t0 = time.time()
        with open(fpath, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith("1 "):
                    try:
                        nid = int(line[2:7].strip())
                        counts[nid] += 1
                    except ValueError:
                        pass
        print(f"    done ({time.time() - t0:.1f}s)")

    print(f"  Scanned {sum(counts.values()):,} TLEs across {len(counts):,} satellites")
    return dict(counts), files


def extract_for_targets(
    files: list[Path],
    target_ids: set[int],
) -> dict[int, list[dict]]:
    """Pass 2: extract full TLE data only for target NORAD IDs."""
    satellites: dict[int, list[dict]] = defaultdict(list)
    total = 0

    for fpath in files:
        print(f"  Extracting from {fpath.name}...")
        t0 = time.time()
        file_count = 0

        with open(fpath, 'r', errors='ignore') as f:
            lines_buf = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines_buf.append(line)
                if len(lines_buf) < 2:
                    continue

                line1, line2 = lines_buf[-2], lines_buf[-1]
                if not line1.startswith("1 ") or not line2.startswith("2 "):
                    continue

                try:
                    nid = int(line1[2:7].strip())
                except ValueError:
                    continue

                if nid not in target_ids:
                    continue

                parsed = parse_tle_lines(line1, line2)
                if parsed:
                    parsed["line1"] = line1
                    parsed["line2"] = line2
                    satellites[nid].append(parsed)
                    total += 1
                    file_count += 1

        print(f"    {file_count:,} TLEs in {time.time() - t0:.1f}s")

    # Sort by epoch
    for nid in satellites:
        satellites[nid].sort(key=lambda r: (r["epoch_year"], r["epoch_day"]))

    print(f"  Total: {total:,} TLEs for {len(satellites):,} target satellites")
    return dict(satellites)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label Space-Track data with IMM-UKF.",
    )
    parser.add_argument("--tle-dir", type=Path, required=True,
                        help="Directory with pre-extracted TLE text files")
    parser.add_argument("--output", type=Path, default=Path("data/imm_ukf_labeled"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-tles", type=int, default=20,
                        help="Min TLEs per satellite for IMM-UKF")
    parser.add_argument("--max-satellites", type=int, default=0)
    parser.add_argument("--max-tles-per-sat", type=int, default=0,
                        help="Max TLEs per satellite (0 = all, keeps most recent)")
    parser.add_argument("--years", help="Comma-separated years (e.g. 2024,2025)")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    year_filter = set(args.years.split(",")) if args.years else None

    # Pass 1: Scan to find top satellites by TLE count (low memory)
    print("=" * 60)
    print("Phase 1a: Scanning NORAD ID counts")
    print("=" * 60)
    t0 = time.time()
    counts, files = _scan_norad_counts(args.tle_dir, year_filter)
    print(f"Scan: {time.time() - t0:.1f}s")

    # Select top satellites
    eligible = [(nid, c) for nid, c in counts.items() if c >= args.min_tles]
    eligible.sort(key=lambda x: -x[1])
    if args.max_satellites:
        eligible = eligible[:args.max_satellites]

    target_ids = {nid for nid, _ in eligible}
    total_target_tles = sum(c for _, c in eligible)
    print(f"Selected {len(target_ids):,} satellites ({total_target_tles:,} TLEs)")

    # Pass 2: Extract full data for targets only
    print(f"\n{'=' * 60}")
    print("Phase 1b: Extracting target satellite TLEs")
    print("=" * 60)
    t0 = time.time()
    satellites = extract_for_targets(files, target_ids)
    print(f"Extraction: {time.time() - t0:.1f}s")

    # Truncate per-satellite TLEs (keep most recent)
    if args.max_tles_per_sat:
        for nid in satellites:
            if len(satellites[nid]) > args.max_tles_per_sat:
                satellites[nid] = satellites[nid][-args.max_tles_per_sat:]

    candidates = sorted(
        [(nid, tles) for nid, tles in satellites.items()],
        key=lambda x: -len(x[1]),
    )
    del satellites  # free memory

    print(f"\n{'=' * 60}")
    print(f"Phase 2: IMM-UKF on {len(candidates):,} satellites "
          f"({sum(len(t) for _, t in candidates):,} TLEs)")
    print(f"{'=' * 60}")

    # Process
    all_results = {}
    total_timesteps = 0
    agree = 0
    disagree = 0
    t0 = time.time()

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, (nid, results) in enumerate(
                pool.imap_unordered(_worker, candidates, chunksize=5)
            ):
                if results:
                    all_results[nid] = results
                    total_timesteps += len(results)
                    for r in results:
                        if r["rule_v1"] == r["imm_ukf"]:
                            agree += 1
                        elif r["rule_v1"] != 0 or r["imm_ukf"] != 0:
                            disagree += 1

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    remaining = (len(candidates) - i - 1) / rate
                    print(f"  [{i+1:,}/{len(candidates):,}] "
                          f"{rate:.1f} sat/s, ~{remaining/60:.0f}min left, "
                          f"labeled={total_timesteps:,}")
    else:
        for i, (nid, tles) in enumerate(candidates):
            results = classify_from_tles(tles, nid)
            if results:
                all_results[nid] = results
                total_timesteps += len(results)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:,}/{len(candidates):,}] "
                      f"labeled={total_timesteps:,}, {elapsed:.0f}s")

    # Save
    import pickle
    out_pkl = args.output / "imm_ukf_labels.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(all_results, f)

    # Also save as flat numpy for training
    all_rv1 = []
    all_imm = []
    all_probs = []
    for nid, results in all_results.items():
        for r in results:
            all_rv1.append(r["rule_v1"])
            all_imm.append(r["imm_ukf"])
            all_probs.append(r["imm_probs"])

    np.save(args.output / "labels_rule_v1.npy", np.array(all_rv1, dtype=np.int32))
    np.save(args.output / "labels_imm_ukf.npy", np.array(all_imm, dtype=np.int32))
    np.save(args.output / "imm_probs.npy", np.array(all_probs, dtype=np.float32))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Satellites labeled: {len(all_results):,}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Time: {time.time() - t0:.1f}s")

    if agree + disagree > 0:
        total_compared = agree + disagree
        print(f"\nrule_v1 vs IMM-UKF (on non-trivial cases):")
        print(f"  Agreement: {agree:,} ({agree/total_compared*100:.1f}%)")
        print(f"  Disagreement: {disagree:,} ({disagree/total_compared*100:.1f}%)")

    # Per-label counts
    from collections import Counter
    rv1_counts = Counter(all_rv1)
    imm_counts = Counter(all_imm)
    label_names = {0: "normal", 1: "maneuver", 2: "decay", 3: "breakup"}
    print(f"\nrule_v1 distribution:")
    for k in sorted(rv1_counts):
        print(f"  {label_names.get(k, k):>12s}: {rv1_counts[k]:>10,}")
    print(f"IMM-UKF distribution:")
    for k in sorted(imm_counts):
        print(f"  {label_names.get(k, k):>12s}: {imm_counts[k]:>10,}")

    print(f"\nSaved to {args.output}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
