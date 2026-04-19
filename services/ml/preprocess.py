"""Data preprocessing — convert raw sources into training-ready sequences.

Handles both synthetic data (already in numpy) and Space-Track JSON
(needs parsing + sequencing). Outputs normalized numpy arrays split
into train/val/test ready for the DataLoader.

Usage:
    # From synthetic data
    python -m services.ml.preprocess --source synthetic --input data/synthetic/ --output data/ml_ready/

    # From Space-Track history
    python -m services.ml.preprocess --source spacetrack --input data/spacetrack/ --output data/ml_ready/
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

# Feature columns: [epoch_h, mean_motion, eccentricity, inclination, bstar, alt_km]
FEATURE_NAMES = ["epoch_h", "mean_motion", "eccentricity", "inclination", "bstar", "alt_km"]
N_FEATURES = len(FEATURE_NAMES)

# Normalization constants (empirical from Starlink data)
FEATURE_MEANS = np.array([0.0, 15.1, 0.0005, 53.0, 0.0, 550.0])
FEATURE_STDS = np.array([100.0, 0.5, 0.001, 20.0, 0.005, 100.0])


def normalize(X: np.ndarray) -> np.ndarray:
    """Z-score normalize features."""
    return (X - FEATURE_MEANS) / np.maximum(FEATURE_STDS, 1e-10)


def denormalize(X: np.ndarray) -> np.ndarray:
    """Reverse z-score normalization."""
    return X * FEATURE_STDS + FEATURE_MEANS


def load_synthetic(input_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load synthetic dataset (already in numpy format)."""
    X = np.load(input_dir / "X.npy")
    y = np.load(input_dir / "y.npy")
    return X, y


def load_spacetrack(input_dir: Path, seq_len: int = 100,
                    stride: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Load Space-Track JSON files and convert to sequences.

    Each JSON file is one satellite's full TLE history. We slide a
    window of seq_len over the history with stride overlap, producing
    multiple training sequences per satellite.

    Labels are all 0 (normal) since we don't have ground truth for
    historical data — the model learns normal dynamics in self-supervised
    mode and anomalies emerge as high prediction error.
    """
    json_files = sorted(input_dir.glob("*.json.gz"))
    if not json_files:
        print(f"No .json.gz files in {input_dir}", file=sys.stderr)
        return np.array([]), np.array([])

    all_X = []
    all_y = []
    skipped = 0

    for i, fpath in enumerate(json_files):
        try:
            with gzip.open(fpath, "rt") as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        if len(records) < seq_len:
            skipped += 1
            continue

        # Extract features
        elements = []
        for r in records:
            try:
                mm = float(r.get("MEAN_MOTION") or 0)
                ecc = float(r.get("ECCENTRICITY") or 0)
                incl = float(r.get("INCLINATION") or 0)
                bstar = float(r.get("BSTAR") or 0)
                sma = float(r.get("SEMIMAJOR_AXIS") or 0)
                alt_km = sma - 6378.137 if sma > 6378 else 0

                epoch_str = r.get("EPOCH", "")
                # Convert epoch to hours from first record
                elements.append([0.0, mm, ecc, incl, bstar, alt_km])
            except (ValueError, TypeError):
                continue

        if len(elements) < seq_len:
            skipped += 1
            continue

        arr = np.array(elements)
        # Fill in relative epoch (hours from start)
        # Approximate: assume ~8h between TLE updates
        arr[:, 0] = np.arange(len(arr)) * 8.0

        # Sliding window
        for start in range(0, len(arr) - seq_len + 1, stride):
            window = arr[start:start + seq_len].copy()
            # Make epoch relative to window start
            window[:, 0] -= window[0, 0]
            all_X.append(window)
            all_y.append(np.zeros(seq_len, dtype=np.int32))

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(json_files)}] sequences so far: {len(all_X)}")

    if skipped:
        print(f"  skipped {skipped} files (too short or corrupt)")

    if not all_X:
        return np.array([]), np.array([])

    return np.array(all_X), np.array(all_y)


def split_dataset(X: np.ndarray, y: np.ndarray,
                  train_frac: float = 0.8, val_frac: float = 0.1,
                  seed: int = 42) -> dict:
    """Split into train/val/test."""
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "spacetrack"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/ml_ready"))
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--normalize", action="store_true", default=True)
    args = parser.parse_args(argv)

    print(f"Loading {args.source} from {args.input}...")
    if args.source == "synthetic":
        X, y = load_synthetic(args.input)
    else:
        X, y = load_spacetrack(args.input, seq_len=args.seq_len)

    if len(X) == 0:
        print("No data loaded!", file=sys.stderr)
        return 1

    print(f"Loaded: X={X.shape}, y={y.shape}")

    if args.normalize:
        X = normalize(X)
        print("Normalized")

    splits = split_dataset(X, y)
    args.output.mkdir(parents=True, exist_ok=True)

    for key, arr in splits.items():
        np.save(args.output / f"{key}.npy", arr)

    print(f"\nSaved to {args.output}/:")
    for key, arr in splits.items():
        print(f"  {key}: {arr.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
