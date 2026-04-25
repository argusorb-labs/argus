#!/usr/bin/env python3
"""Build training dataset by fusing IMM-UKF labels with TLE features.

Takes IMM-UKF labeled results (pkl) + raw TLE text files,
reconstructs full feature sequences, and outputs training-ready numpy.

The fusion strategy:
  - Where IMM-UKF has a label → use IMM-UKF (higher quality)
  - IMM-UKF labels: 0=normal, 1=maneuver, 2=decay
  - Also stores rule_v1 label for comparison

Usage:
    python scripts/build_imm_training_set.py \
        --imm-pkl data/imm_ukf_labeled_200/imm_ukf_labels.pkl \
        --tle-dir data/spacetrack_raw_tle/ \
        --output data/ml_imm_fused/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.label_spacetrack_bulk import BULK_FEATURE_MEANS, BULK_FEATURE_STDS
from scripts.parse_spacetrack_zip import parse_tle_lines


def load_imm_results(pkl_path: Path) -> dict[int, list[dict]]:
    """Load IMM-UKF results from pickle."""
    import pickle
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def extract_satellite_features(
    tle_dir: Path,
    target_ids: set[int],
    max_tles_per_sat: int = 2000,
) -> dict[int, np.ndarray]:
    """Extract full orbital element arrays for target satellites.

    Returns: {norad_id: (N, 6) array of [epoch_h, mm, ecc, incl, bstar, alt_km]}
    """
    satellites: dict[int, list[dict]] = defaultdict(list)

    for fpath in sorted(tle_dir.glob("*.txt")):
        print(f"  Reading {fpath.name}...")
        with open(fpath, 'r', errors='ignore') as f:
            prev_line = ""
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if prev_line.startswith("1 ") and line.startswith("2 "):
                    try:
                        nid = int(prev_line[2:7].strip())
                    except ValueError:
                        prev_line = line
                        continue

                    if nid in target_ids:
                        parsed = parse_tle_lines(prev_line, line)
                        if parsed:
                            satellites[nid].append(parsed)

                prev_line = line

    # Sort by epoch and truncate
    result = {}
    for nid in satellites:
        records = sorted(satellites[nid],
                         key=lambda r: (r["epoch_year"], r["epoch_day"]))
        if len(records) > max_tles_per_sat:
            records = records[-max_tles_per_sat:]

        n = len(records)
        arr = np.zeros((n, 11), dtype=np.float64)

        def _epoch_hours(r):
            return (r["epoch_year"] * 365.25 + r["epoch_day"]) * 24.0

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

        # dt_hours: real time since previous TLE
        arr[0, 6] = 0.0
        arr[1:, 6] = np.diff(epoch_h)
        arr[:, 6] = np.clip(arr[:, 6], 0.0, 240.0)

        result[nid] = arr

    print(f"  Extracted features for {len(result):,} satellites")
    return result


def build_sequences(
    features: dict[int, np.ndarray],
    imm_results: dict[int, list[dict]],
    seq_len: int = 50,
    stride: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window sequences with IMM-UKF labels.

    Returns: (X, y_imm, y_rv1, soft_labels) all as numpy arrays.
    X: (N, seq_len, 11) normalized features
    y_imm: (N, seq_len) IMM-UKF hard labels
    y_rv1: (N, seq_len) rule_v1 labels (for comparison)
    soft_labels: (N, seq_len, 3) IMM model probabilities [P(sk), P(man), P(dec)]
    """
    means = np.array(BULK_FEATURE_MEANS, dtype=np.float32)
    stds = np.maximum(np.array(BULK_FEATURE_STDS, dtype=np.float32), 1e-10)

    all_X = []
    all_y_imm = []
    all_y_rv1 = []
    all_soft = []

    for nid, feat_arr in features.items():
        if nid not in imm_results:
            continue

        imm_recs = imm_results[nid]
        n_tles = len(feat_arr)

        if len(imm_recs) != n_tles:
            n = min(len(imm_recs), n_tles)
            feat_arr = feat_arr[-n:]
            imm_recs = imm_recs[-n:]

        y_imm_full = np.array([r["imm_ukf"] for r in imm_recs], dtype=np.int32)
        y_rv1_full = np.array([r["rule_v1"] for r in imm_recs], dtype=np.int32)
        # Soft labels: IMM model probabilities [P(normal), P(maneuver), P(decay)]
        soft_full = np.array([r["imm_probs"] for r in imm_recs], dtype=np.float32)

        if n_tles < seq_len:
            continue

        for start in range(0, n_tles - seq_len + 1, stride):
            window = feat_arr[start:start + seq_len].copy()
            window[:, 0] -= window[0, 0]
            window[0, 6] = 0.0

            window[:, 4] = np.clip(window[:, 4], -1.0, 1.0)
            window_norm = ((window - means) / stds).astype(np.float32)

            all_X.append(window_norm)
            all_y_imm.append(y_imm_full[start:start + seq_len])
            all_y_rv1.append(y_rv1_full[start:start + seq_len])
            all_soft.append(soft_full[start:start + seq_len])

    X = np.array(all_X)
    y_imm = np.array(all_y_imm)
    y_rv1 = np.array(all_y_rv1)
    soft = np.array(all_soft)

    return X, y_imm, y_rv1, soft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build training set fusing IMM-UKF labels with TLE features.",
    )
    parser.add_argument("--imm-pkl", type=Path, required=True)
    parser.add_argument("--tle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/ml_imm_fused"))
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-tles-per-sat", type=int, default=2000)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    # Load IMM-UKF results
    print("Loading IMM-UKF results...")
    imm_results = load_imm_results(args.imm_pkl)
    target_ids = set(imm_results.keys())
    print(f"  {len(target_ids)} satellites with IMM-UKF labels")

    # Extract features from raw TLE files
    print("\nExtracting TLE features...")
    t0 = time.time()
    features = extract_satellite_features(
        args.tle_dir, target_ids, max_tles_per_sat=args.max_tles_per_sat
    )
    print(f"  {time.time() - t0:.1f}s")

    # Build sequences
    print("\nBuilding sequences...")
    X, y_imm, y_rv1, soft = build_sequences(
        features, imm_results, seq_len=args.seq_len, stride=args.stride
    )
    print(f"  {len(X):,} sequences, shape {X.shape}")

    # Split: 80/10/10
    rng = np.random.default_rng(42)
    n = len(X)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    # Save — use IMM-UKF labels as primary (y_train), rule_v1 as secondary
    np.save(args.output / "X_train.npy", X[idx[:n_train]])
    np.save(args.output / "y_train.npy", y_imm[idx[:n_train]])
    np.save(args.output / "X_val.npy", X[idx[n_train:n_train + n_val]])
    np.save(args.output / "y_val.npy", y_imm[idx[n_train:n_train + n_val]])
    np.save(args.output / "X_test.npy", X[idx[n_train + n_val:]])
    np.save(args.output / "y_test.npy", y_imm[idx[n_train + n_val:]])

    # Also save rule_v1 labels and soft labels
    np.save(args.output / "y_train_rv1.npy", y_rv1[idx[:n_train]])
    np.save(args.output / "y_test_rv1.npy", y_rv1[idx[n_train + n_val:]])
    np.save(args.output / "soft_train.npy", soft[idx[:n_train]])
    np.save(args.output / "soft_val.npy", soft[idx[n_train:n_train + n_val]])
    np.save(args.output / "soft_test.npy", soft[idx[n_train + n_val:]])

    # Summary
    from collections import Counter
    imm_counts = Counter(y_imm.ravel().tolist())
    rv1_counts = Counter(y_rv1.ravel().tolist())
    label_names = {0: "normal", 1: "maneuver", 2: "decay", 3: "breakup"}

    print(f"\n{'=' * 50}")
    print(f"Total: {n:,} sequences")
    print(f"Train: {n_train:,}  Val: {n_val:,}  Test: {n - n_train - n_val:,}")
    print(f"\nIMM-UKF label distribution:")
    for k in sorted(imm_counts):
        print(f"  {label_names.get(k, k):>10s}: {imm_counts[k]:>10,} "
              f"({imm_counts[k] / y_imm.size * 100:.2f}%)")
    print(f"\nrule_v1 label distribution:")
    for k in sorted(rv1_counts):
        print(f"  {label_names.get(k, k):>10s}: {rv1_counts[k]:>10,}")
    print(f"\nSaved to {args.output}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
