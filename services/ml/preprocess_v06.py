"""v0.6 preprocessor: extends v0.5 with pre-computed innovation features.

The v0.5 AnalyticalPhysics baseline was too crude — innovation ended up
dominated by physics model error rather than anomaly signal, giving a
45% false-positive rate on normal trajectories. v0.6 fixes this by:

1. Converting each TLE to a Cartesian state via sgp4 (ground truth pos/vel).
2. Propagating state_{t-1} forward to epoch_t using the full J2+J3+J4+drag
   integrator (services/brain/dynamics).
3. Taking innovation_t = state_t − predicted_state_t.

Innovation is included as 6 additional feature channels. Model input goes
from 6 to 12 features. Physics is now *truly* frozen — computed offline,
not as an in-model learnable module.

Usage:
    python -m services.ml.preprocess_v06 \
        --input data/spacetrack \
        --output data/ml_ready_v06 \
        --seq-len 100 --stride 50
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from services.ml.physics import compute_innovation_series_sgp4

# Feature columns: 6 TLE elements + 6 Cartesian innovation channels
FEATURE_NAMES = [
    # Original TLE elements
    "epoch_h",
    "mean_motion",
    "eccentricity",
    "inclination",
    "bstar",
    "alt_km",
    # Innovation in TEME (meters for pos, m/s for vel)
    "inn_dx",
    "inn_dy",
    "inn_dz",
    "inn_dvx",
    "inn_dvy",
    "inn_dvz",
]
N_FEATURES = len(FEATURE_NAMES)

# Normalization — innovation position is in meters (km scale),
# velocity is m/s. Magnitudes chosen so anomaly signatures fall
# roughly in the [-3, +3] normalized range.
FEATURE_MEANS = np.array(
    [0.0, 15.1, 0.0005, 53.0, 0.0, 550.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
FEATURE_STDS = np.array(
    [100.0, 0.5, 0.001, 20.0, 0.005, 100.0, 2000.0, 2000.0, 2000.0, 2.0, 2.0, 2.0]
)


def normalize(X: np.ndarray) -> np.ndarray:
    return (X - FEATURE_MEANS) / np.maximum(FEATURE_STDS, 1e-10)


def parse_epoch_string(s: str) -> float:
    """Space-Track EPOCH → Unix seconds. Accepts ISO variants."""
    s = s.replace("Z", "").strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unrecognized EPOCH format: {s!r}")


def _load_one_satellite(records: list[dict]) -> np.ndarray | None:
    """Extract (T, 12) feature array for one satellite's TLE history.

    Returns None if fewer than 2 valid TLEs (can't compute any innovation).
    """
    parsed = []
    for r in records:
        l1 = r.get("TLE_LINE1")
        l2 = r.get("TLE_LINE2")
        if not l1 or not l2:
            continue
        try:
            mm = float(r["MEAN_MOTION"])
            ecc = float(r["ECCENTRICITY"])
            incl = float(r["INCLINATION"])
            bstar = float(r["BSTAR"])
            sma = float(r["SEMIMAJOR_AXIS"])
            alt_km = sma - 6378.137 if sma > 6378 else 0.0
            epoch_ts = parse_epoch_string(r["EPOCH"])
        except (KeyError, ValueError, TypeError):
            continue

        parsed.append(
            {
                "epoch_ts": epoch_ts,
                "l1": l1,
                "l2": l2,
                "mm": mm,
                "ecc": ecc,
                "incl": incl,
                "bstar": bstar,
                "alt_km": alt_km,
            }
        )

    if len(parsed) < 2:
        return None

    # Sort chronologically (gp_history is usually ASC but be safe)
    parsed.sort(key=lambda d: d["epoch_ts"])
    T = len(parsed)

    # SGP4-based innovation: for each t>=1, innovation[t] =
    # SGP4(TLE[t], epoch_t) - SGP4(TLE[t-1], epoch_t)
    l1s = [p["l1"] for p in parsed]
    l2s = [p["l2"] for p in parsed]
    innovations = compute_innovation_series_sgp4(l1s, l2s)  # (T, 6)

    # Build feature matrix
    features = np.zeros((T, N_FEATURES))
    t0 = parsed[0]["epoch_ts"]
    for t in range(T):
        p = parsed[t]
        features[t, 0] = (p["epoch_ts"] - t0) / 3600.0
        features[t, 1] = p["mm"]
        features[t, 2] = p["ecc"]
        features[t, 3] = p["incl"]
        features[t, 4] = p["bstar"]
        features[t, 5] = p["alt_km"]
        features[t, 6:12] = innovations[t]

    return features


def load_spacetrack(
    input_dir: Path,
    seq_len: int = 100,
    stride: int = 50,
    max_files: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all Space-Track JSON files and produce (N, seq_len, 12) sequences."""
    json_files = sorted(input_dir.glob("*.json.gz"))
    if not json_files:
        print(f"No .json.gz in {input_dir}", file=sys.stderr)
        return np.array([]), np.array([])
    if max_files > 0:
        json_files = json_files[:max_files]

    all_X = []
    all_y = []
    skipped = 0
    total_tle = 0

    for i, fpath in enumerate(json_files):
        try:
            with gzip.open(fpath, "rt") as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        # Skip rate-limit placeholders (already cleaned once but defensive)
        if (
            isinstance(records, list)
            and records
            and isinstance(records[0], dict)
            and "error" in records[0]
        ):
            skipped += 1
            continue

        if len(records) < seq_len:
            skipped += 1
            continue

        features = _load_one_satellite(records)
        if features is None or len(features) < seq_len:
            skipped += 1
            continue

        total_tle += len(features)

        # Sliding window
        for start in range(0, len(features) - seq_len + 1, stride):
            window = features[start : start + seq_len].copy()
            # Make epoch relative to window start
            window[:, 0] -= window[0, 0]
            all_X.append(window)
            all_y.append(np.zeros(seq_len, dtype=np.int32))

        if (i + 1) % 50 == 0:
            print(
                f"  [{i + 1}/{len(json_files)}] sequences: {len(all_X):,}, "
                f"TLEs: {total_tle:,}, skipped: {skipped}"
            )

    print(
        f"Done loading: {len(all_X):,} sequences from {total_tle:,} TLEs ({skipped} files skipped)"
    )

    if not all_X:
        return np.array([]), np.array([])

    return np.array(all_X), np.array(all_y)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "X_train": X[indices[:n_train]],
        "y_train": y[indices[:n_train]],
        "X_val": X[indices[n_train : n_train + n_val]],
        "y_val": y[indices[n_train : n_train + n_val]],
        "X_test": X[indices[n_train + n_val :]],
        "y_test": y[indices[n_train + n_val :]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/spacetrack"))
    parser.add_argument("--output", type=Path, default=Path("data/ml_ready_v06"))
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Cap input files (0 = all). Useful for dry-run smoke test.",
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip z-score normalization"
    )
    args = parser.parse_args(argv)

    print(f"Loading Space-Track from {args.input} (max_files={args.max_files})...")
    X, y = load_spacetrack(
        args.input, seq_len=args.seq_len, stride=args.stride, max_files=args.max_files
    )
    if len(X) == 0:
        print("No data loaded!", file=sys.stderr)
        return 1

    print(f"Raw: X={X.shape}, y={y.shape}")
    print("Innovation stats (meters):")
    inn_mag = np.linalg.norm(X[:, :, 6:9], axis=2)  # position innovation mag
    print(
        f"  position innovation — median {np.median(inn_mag):.1f} m, "
        f"p99 {np.percentile(inn_mag, 99):.1f} m, max {inn_mag.max():.0f} m"
    )

    if not args.no_normalize:
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
