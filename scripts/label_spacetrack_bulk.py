#!/usr/bin/env python3
"""Label all parsed Space-Track data with rule_v1.

Reads per-year numpy files from parse_spacetrack_zip.py output,
runs rule_v1 on every consecutive TLE pair within each sequence,
and outputs labeled training data (X + y) ready for the ML pipeline.

This gives us MILLIONS of labels on REAL data — not synthetic.
Labels are "noisy" (rule_v1 has false positives), but the data itself
is real TLE with real NORAD noise patterns. Much better than perfect
labels on fake data.

Usage:
    python scripts/label_spacetrack_bulk.py \
        --input data/spacetrack_parsed/ \
        --output data/ml_ready_labeled/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Normalization stats for ALL Space-Track satellites (not just Starlink).
# Computed from a 29k-sequence sample across all 29 year files.
# B* is clipped to [-1, 1] before normalization (raw values have extreme outliers).
BULK_FEATURE_MEANS = [196.0, 12.1633, 0.0616, 73.9842, 0.0, 4240.44]
BULK_FEATURE_STDS = [115.45, 4.2627, 0.1683, 27.267, 0.01, 8881.41]

# rule_v1 thresholds (same as orbital_analyzer.py)
REENTRY_ALT_KM = 250.0
LOW_ORBIT_ALT_KM = 400.0
DECAY_DELTA_KM = 5.0
MANEUVER_ALT_DELTA_KM = 10.0
INCLINATION_DELTA_DEG = 0.1
ECC_DELTA = 0.01
BSTAR_SIGN_FLIP_FLOOR = 5e-3
BSTAR_JUMP_RATIO = 2.0
BSTAR_JUMP_ABS_MIN = 5e-3

# Labels
NORMAL = 0
MANEUVER = 1
DECAY = 2
BREAKUP = 3  # reentry also maps here


def label_pair(old: np.ndarray, new: np.ndarray) -> int:
    """Apply rule_v1 logic to a pair of raw (unnormalized) TLE elements.

    Elements: [epoch_h, mean_motion, eccentricity, inclination, bstar, alt_km]
    Returns: label (0=normal, 1=maneuver, 2=decay, 3=breakup/reentry)
    """
    mm_old, mm_new = old[1], new[1]
    ecc_old, ecc_new = old[2], new[2]
    incl_old, incl_new = old[3], new[3]
    bstar_old, bstar_new = old[4], new[4]
    alt_old, alt_new = old[5], new[5]

    if mm_old <= 0 or mm_new <= 0:
        return NORMAL

    delta_alt = alt_new - alt_old
    delta_ecc = abs(ecc_new - ecc_old)
    delta_incl = abs(incl_new - incl_old)

    # 1. Reentry
    if alt_new < REENTRY_ALT_KM:
        return BREAKUP

    # 2. Decay
    if delta_alt < -DECAY_DELTA_KM and alt_new < LOW_ORBIT_ALT_KM:
        return DECAY

    # 3. Inclination shift
    if delta_incl > INCLINATION_DELTA_DEG:
        return MANEUVER

    # 4. Altitude change
    if abs(delta_alt) > MANEUVER_ALT_DELTA_KM:
        return MANEUVER

    # 5. Eccentricity change
    if delta_ecc > ECC_DELTA:
        return MANEUVER

    # 6. B* sign flip
    if (abs(bstar_old) > BSTAR_SIGN_FLIP_FLOOR
            and abs(bstar_new) > BSTAR_SIGN_FLIP_FLOOR
            and (bstar_old > 0) != (bstar_new > 0)):
        return MANEUVER

    # 7. B* magnitude jump
    delta_bstar = abs(bstar_new - bstar_old)
    denom = max(abs(bstar_old), BSTAR_SIGN_FLIP_FLOOR)
    if delta_bstar > BSTAR_JUMP_ABS_MIN and delta_bstar / denom > BSTAR_JUMP_RATIO:
        return DECAY  # atmospheric anomaly maps to decay

    return NORMAL


def label_sequences(X_raw: np.ndarray) -> np.ndarray:
    """Label all sequences using rule_v1 on consecutive pairs (vectorized).

    Args:
        X_raw: (N, T, 6) UNnormalized sequences
               Features: [epoch_h, mean_motion, eccentricity, inclination, bstar, alt_km]

    Returns:
        y: (N, T) labels
    """
    N, T, F = X_raw.shape
    y = np.zeros((N, T), dtype=np.int32)

    # Extract features: old = t-1, new = t
    old = X_raw[:, :-1, :]  # (N, T-1, 6)
    new = X_raw[:, 1:, :]   # (N, T-1, 6)

    mm_old, mm_new = old[:, :, 1], new[:, :, 1]
    ecc_old, ecc_new = old[:, :, 2], new[:, :, 2]
    incl_old, incl_new = old[:, :, 3], new[:, :, 3]
    bstar_old, bstar_new = old[:, :, 4], new[:, :, 4]
    alt_old, alt_new = old[:, :, 5], new[:, :, 5]

    delta_alt = alt_new - alt_old
    delta_ecc = np.abs(ecc_new - ecc_old)
    delta_incl = np.abs(incl_new - incl_old)

    # Skip invalid (mean_motion <= 0)
    valid = (mm_old > 0) & (mm_new > 0)

    # Apply rules in REVERSE priority (later rules overwrite earlier)
    labels = np.zeros((N, T - 1), dtype=np.int32)

    # 7. B* magnitude jump → DECAY
    delta_bstar = np.abs(bstar_new - bstar_old)
    denom = np.maximum(np.abs(bstar_old), BSTAR_SIGN_FLIP_FLOOR)
    m7 = valid & (delta_bstar > BSTAR_JUMP_ABS_MIN) & (delta_bstar / denom > BSTAR_JUMP_RATIO)
    labels[m7] = DECAY

    # 6. B* sign flip → MANEUVER
    m6 = valid & (np.abs(bstar_old) > BSTAR_SIGN_FLIP_FLOOR) & \
         (np.abs(bstar_new) > BSTAR_SIGN_FLIP_FLOOR) & \
         ((bstar_old > 0) != (bstar_new > 0))
    labels[m6] = MANEUVER

    # 5. Eccentricity change → MANEUVER
    m5 = valid & (delta_ecc > ECC_DELTA)
    labels[m5] = MANEUVER

    # 4. Altitude change → MANEUVER
    m4 = valid & (np.abs(delta_alt) > MANEUVER_ALT_DELTA_KM)
    labels[m4] = MANEUVER

    # 3. Inclination shift → MANEUVER
    m3 = valid & (delta_incl > INCLINATION_DELTA_DEG)
    labels[m3] = MANEUVER

    # 2. Decay
    m2 = valid & (delta_alt < -DECAY_DELTA_KM) & (alt_new < LOW_ORBIT_ALT_KM)
    labels[m2] = DECAY

    # 1. Reentry → BREAKUP (highest priority)
    m1 = valid & (alt_new < REENTRY_ALT_KM)
    labels[m1] = BREAKUP

    y[:, 1:] = labels
    return y


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Label Space-Track data with rule_v1 for ML training.",
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory with X_*.npy files from parse_spacetrack_zip.py")
    parser.add_argument("--output", type=Path, default=Path("data/ml_ready_labeled"))
    parser.add_argument("--max-sequences", type=int, default=0,
                        help="Limit total sequences (0 = all)")
    parser.add_argument("--years", help="Comma-separated years (e.g. 2024,2025)")
    args = parser.parse_args(argv)

    # Find per-year numpy files
    files = sorted(args.input.glob("X_tle*.npy"))
    if args.years:
        year_filter = set(args.years.split(","))
        files = [f for f in files if any(y in f.name for y in year_filter)]

    if not files:
        print(f"No X_tle*.npy files in {args.input}")
        return 1

    print(f"Found {len(files)} year files to label")
    args.output.mkdir(parents=True, exist_ok=True)

    # Phase 1: Label each file individually and save labels
    # This avoids loading all 19 GB into RAM at once
    label_dir = args.output / "per_file"
    label_dir.mkdir(parents=True, exist_ok=True)

    file_sizes = []  # (file_path, label_path, n_sequences)
    total_labeled = 0
    total_anomalies = 0

    for fpath in files:
        label_path = label_dir / f"y_{fpath.stem}.npy"
        print(f"\n[{fpath.name}]")
        t0 = time.time()

        X_raw = np.load(fpath)
        print(f"  loaded: {X_raw.shape}")

        # Label
        y = label_sequences(X_raw)
        n_anomalies = int((y != 0).sum())
        anomaly_pct = n_anomalies / y.size * 100

        np.save(label_path, y)
        file_sizes.append((fpath, label_path, len(X_raw)))

        total_labeled += y.size
        total_anomalies += n_anomalies

        print(f"  labeled: {n_anomalies:,} anomaly steps ({anomaly_pct:.2f}%), "
              f"{time.time() - t0:.1f}s")

        del X_raw, y  # free RAM

        if args.max_sequences and sum(s for _, _, s in file_sizes) >= args.max_sequences:
            print(f"  reached max_sequences={args.max_sequences}")
            break

    # Phase 2: Build split assignment array
    # Instead of loading everything into RAM, assign each global index to a split
    # 0=train, 1=val, 2=test — then stream files and write by split
    total_seqs = sum(s for _, _, s in file_sizes)
    if args.max_sequences and total_seqs > args.max_sequences:
        total_seqs = args.max_sequences

    print(f"\nBuilding split assignment for {total_seqs:,} sequences...")
    rng = np.random.default_rng(42)

    n_train = int(total_seqs * 0.8)
    n_val = int(total_seqs * 0.1)
    n_test = total_seqs - n_train - n_val

    # Create assignment array: shuffle indices, first 80% → train, next 10% → val, rest → test
    split_assign = np.empty(total_seqs, dtype=np.int8)
    perm = rng.permutation(total_seqs)
    split_assign[perm[:n_train]] = 0
    split_assign[perm[n_train:n_train + n_val]] = 1
    split_assign[perm[n_train + n_val:]] = 2
    del perm

    # Phase 3: Stream through files and write to splits (vectorized)
    seq_len = np.load(file_sizes[0][0], mmap_mode='r').shape[1]
    n_feat = np.load(file_sizes[0][0], mmap_mode='r').shape[2]

    print(f"Allocating output arrays: train={n_train:,} val={n_val:,} test={n_test:,}")
    X_train = np.lib.format.open_memmap(
        str(args.output / "X_train.npy"), mode='w+',
        dtype=np.float32, shape=(n_train, seq_len, n_feat))
    y_train = np.lib.format.open_memmap(
        str(args.output / "y_train.npy"), mode='w+',
        dtype=np.int32, shape=(n_train, seq_len))
    X_val = np.lib.format.open_memmap(
        str(args.output / "X_val.npy"), mode='w+',
        dtype=np.float32, shape=(n_val, seq_len, n_feat))
    y_val = np.lib.format.open_memmap(
        str(args.output / "y_val.npy"), mode='w+',
        dtype=np.int32, shape=(n_val, seq_len))
    X_test = np.lib.format.open_memmap(
        str(args.output / "X_test.npy"), mode='w+',
        dtype=np.float32, shape=(n_test, seq_len, n_feat))
    y_test = np.lib.format.open_memmap(
        str(args.output / "y_test.npy"), mode='w+',
        dtype=np.int32, shape=(n_test, seq_len))

    split_arrays_X = [X_train, X_val, X_test]
    split_arrays_y = [y_train, y_val, y_test]

    means = np.array(BULK_FEATURE_MEANS, dtype=np.float32)
    stds = np.maximum(np.array(BULK_FEATURE_STDS, dtype=np.float32), 1e-10)

    write_pos = [0, 0, 0]  # current write position for train/val/test
    global_offset = 0

    for fpath, label_path, n_seqs in file_sizes:
        print(f"\n  Streaming {fpath.name}...")
        t0 = time.time()
        X_raw = np.load(fpath)
        y_raw = np.load(label_path)

        # How many sequences from this file are within our total
        use_n = min(n_seqs, total_seqs - global_offset)
        if use_n <= 0:
            break

        # Clip B* outliers before normalization (feature index 4)
        X_clipped = X_raw[:use_n].copy()
        X_clipped[:, :, 4] = np.clip(X_clipped[:, :, 4], -1.0, 1.0)

        # Normalize
        X_norm = ((X_clipped - means) / stds).astype(np.float32)
        del X_clipped
        y_chunk = y_raw[:use_n]
        del X_raw, y_raw

        # Get split assignments for this chunk (vectorized)
        chunk_splits = split_assign[global_offset:global_offset + use_n]

        for s in range(3):
            mask = chunk_splits == s
            count = mask.sum()
            if count == 0:
                continue
            pos = write_pos[s]
            split_arrays_X[s][pos:pos + count] = X_norm[mask]
            split_arrays_y[s][pos:pos + count] = y_chunk[mask]
            write_pos[s] += count

        global_offset += use_n
        del X_norm, y_chunk, chunk_splits
        print(f"    done ({time.time() - t0:.1f}s) — "
              f"train={write_pos[0]:,} val={write_pos[1]:,} test={write_pos[2]:,}")

    # Flush
    del X_train, y_train, X_val, y_val, X_test, y_test

    from collections import Counter
    # Read back labels for summary
    y_all_train = np.load(args.output / "y_train.npy", mmap_mode='r')
    label_counts = Counter(y_all_train.ravel().tolist())

    print(f"\n{'=' * 50}")
    print(f"Total: {total_seqs:,} sequences, {total_labeled:,} timesteps")
    print(f"Train label distribution: {dict(label_counts)}")
    print(f"Anomaly rate: {total_anomalies / total_labeled * 100:.2f}%")
    print(f"Saved to {args.output}/")
    for f in sorted(args.output.glob("*.npy")):
        arr = np.load(f, mmap_mode='r')
        print(f"  {f.name}: {arr.shape}, {arr.nbytes / 1e9:.2f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
