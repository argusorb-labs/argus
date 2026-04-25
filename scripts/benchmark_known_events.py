#!/usr/bin/env python3
"""Benchmark all classifiers on known orbital events.

Evaluates rule_v1, IMM-UKF, and ML models on ground-truth events
where we independently know what happened.

Usage:
    python scripts/benchmark_known_events.py --tle-dir data/spacetrack_raw_tle/
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

from scripts.parse_spacetrack_zip import parse_tle_lines
from scripts.label_spacetrack_bulk import label_pair

# Known events with ground truth
KNOWN_EVENTS = [
    {
        "name": "Starlink-34343 debris",
        "norad_id": 64157,
        "event_epoch_year": 2026,
        "event_epoch_day": 88.8,  # ~March 29
        "expected_class": 3,  # breakup
        "description": "Confirmed debris event by S4S",
    },
    {
        "name": "Iridium 33 collision",
        "norad_id": 24946,
        "event_epoch_year": 2009,
        "event_epoch_day": 41.7,  # Feb 10
        "expected_class": 3,  # breakup/collision
        "description": "First confirmed accidental hypervelocity collision",
    },
    {
        "name": "Cosmos 2251 collision",
        "norad_id": 22675,
        "event_epoch_year": 2009,
        "event_epoch_day": 41.7,  # Feb 10
        "expected_class": 3,  # breakup/collision
        "description": "Collided with Iridium 33",
    },
]


def extract_satellite_tles(tle_dir: Path, norad_id: int) -> list[dict]:
    """Extract all TLEs for a satellite from raw text files."""
    records = []
    for fpath in sorted(tle_dir.glob("*.txt")):
        with open(fpath, 'r', errors='ignore') as f:
            prev = ''
            for line in f:
                line = line.strip()
                if prev.startswith("1 ") and line.startswith("2 "):
                    try:
                        nid = int(prev[2:7].strip())
                    except ValueError:
                        prev = line
                        continue
                    if nid == norad_id:
                        parsed = parse_tle_lines(prev, line)
                        if parsed:
                            parsed["line1"] = prev
                            parsed["line2"] = line
                            records.append(parsed)
                prev = line

    records.sort(key=lambda r: (r["epoch_year"], r["epoch_day"]))
    return records


def build_sequence(records: list[dict], center_idx: int, seq_len: int = 50) -> np.ndarray | None:
    """Build a normalized 11-feature sequence centered around an index."""
    from scripts.label_spacetrack_bulk import BULK_FEATURE_MEANS, BULK_FEATURE_STDS

    start = max(0, center_idx - seq_len + 5)  # 5 TLEs after event in window
    end = start + seq_len
    if end > len(records):
        end = len(records)
        start = max(0, end - seq_len)

    if end - start < seq_len:
        return None

    chunk = records[start:end]
    arr = np.zeros((seq_len, 11), dtype=np.float64)

    def _epoch_hours(r):
        return (r["epoch_year"] * 365.25 + r["epoch_day"]) * 24.0

    epoch_h = np.array([_epoch_hours(r) for r in chunk])

    for j, r in enumerate(chunk):
        arr[j, 0] = epoch_h[j]
        arr[j, 1] = r["mean_motion"]
        arr[j, 2] = r["eccentricity"]
        arr[j, 3] = r["inclination"]
        arr[j, 4] = r["bstar"]
        arr[j, 5] = r["alt_km"]
        arr[j, 7] = r.get("raan", 0)
        arr[j, 8] = r.get("argp", 0)
        arr[j, 9] = r.get("mean_anomaly", 0)
        arr[j, 10] = r.get("n_dot", 0)

    arr[:, 0] -= arr[0, 0]
    arr[0, 6] = 0.0
    arr[1:, 6] = np.diff(epoch_h)
    arr[:, 6] = np.clip(arr[:, 6], 0.0, 240.0)
    arr[:, 4] = np.clip(arr[:, 4], -1.0, 1.0)

    means = np.array(BULK_FEATURE_MEANS, dtype=np.float32)
    stds = np.maximum(np.array(BULK_FEATURE_STDS, dtype=np.float32), 1e-10)
    return ((arr - means) / stds).astype(np.float32)


def eval_rule_v1(records: list[dict], event_idx: int) -> dict:
    """Evaluate rule_v1 around the event."""
    if event_idx < 1 or event_idx >= len(records):
        return {"detected": False, "latency": None, "label": 0}

    # Check TLEs around the event
    for offset in range(0, min(10, len(records) - event_idx)):
        idx = event_idx + offset
        if idx < 1:
            continue
        old = np.array([0, records[idx-1]["mean_motion"], records[idx-1]["eccentricity"],
                        records[idx-1]["inclination"], records[idx-1]["bstar"],
                        records[idx-1]["alt_km"]])
        new = np.array([0, records[idx]["mean_motion"], records[idx]["eccentricity"],
                        records[idx]["inclination"], records[idx]["bstar"],
                        records[idx]["alt_km"]])
        label = label_pair(old, new)
        if label != 0:
            return {"detected": True, "latency": offset, "label": label}

    return {"detected": False, "latency": None, "label": 0}


def eval_ml(records: list[dict], event_idx: int, checkpoint_path: str, size: str = "medium") -> dict:
    """Evaluate ML model around the event."""
    import torch
    from services.ml.model import create_model

    seq = build_sequence(records, event_idx)
    if seq is None:
        return {"detected": False, "latency": None, "label": 0, "probs": None}

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ckpt.get("model_config", {})
    n_features = config.get("n_features", 11)
    model = create_model(size, n_features=n_features)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X = torch.from_numpy(seq[:, :n_features][np.newaxis]).float()
    with torch.no_grad():
        out = model(X, causal=True)
        cls = out["classifications"][0]
        probs = torch.softmax(cls, dim=-1).numpy()
        preds = cls.argmax(dim=-1).numpy()

    # Find first non-normal prediction at or after event position in window
    event_pos_in_window = min(45, event_idx - max(0, event_idx - 45))
    for t in range(event_pos_in_window, len(preds)):
        if preds[t] != 0:
            return {
                "detected": True,
                "latency": t - event_pos_in_window,
                "label": int(preds[t]),
                "probs": probs[t].tolist(),
            }

    return {"detected": False, "latency": None, "label": 0,
            "probs": probs[event_pos_in_window].tolist() if event_pos_in_window < len(probs) else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark on known events.")
    parser.add_argument("--tle-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="*", default=[
        "checkpoints/v11_medium_11feat_finetune/best_model.pt",
        "checkpoints/v11b_final/best_model.pt",
    ])
    args = parser.parse_args(argv)

    label_names = {0: "normal", 1: "maneuver", 2: "decay", 3: "breakup"}

    print("=" * 70)
    print("BENCHMARK: Known Orbital Events")
    print("=" * 70)

    for event in KNOWN_EVENTS:
        nid = event["norad_id"]
        print(f"\n{'─' * 70}")
        print(f"Event: {event['name']} (NORAD {nid})")
        print(f"  {event['description']}")
        print(f"  Expected: {label_names[event['expected_class']]}")

        # Extract TLEs
        records = extract_satellite_tles(args.tle_dir, nid)
        if not records:
            print(f"  No TLE data found for NORAD {nid}")
            continue

        print(f"  TLEs found: {len(records)}")

        # Find event epoch
        event_epoch = event["event_epoch_year"] * 365.25 + event["event_epoch_day"]
        event_idx = None
        for i, r in enumerate(records):
            rec_epoch = r["epoch_year"] * 365.25 + r["epoch_day"]
            if rec_epoch >= event_epoch:
                event_idx = i
                break

        if event_idx is None:
            print(f"  Event epoch not found in TLE data")
            continue

        print(f"  Event at TLE index {event_idx}/{len(records)}")

        # Evaluate rule_v1
        rv1 = eval_rule_v1(records, event_idx)
        detected_str = "YES" if rv1["detected"] else "NO"
        label_str = label_names.get(rv1["label"], "?")
        latency_str = f"+{rv1['latency']} TLEs" if rv1["latency"] is not None else "—"
        print(f"\n  rule_v1:  detected={detected_str}  label={label_str}  latency={latency_str}")

        # Evaluate ML models
        for ckpt_path in args.checkpoints:
            if not os.path.exists(ckpt_path):
                continue
            name = Path(ckpt_path).parent.name
            ml = eval_ml(records, event_idx, ckpt_path)
            detected_str = "YES" if ml["detected"] else "NO"
            label_str = label_names.get(ml["label"], "?")
            latency_str = f"+{ml['latency']} TLEs" if ml["latency"] is not None else "—"
            probs_str = ""
            if ml["probs"]:
                probs_str = f"  P=[{ml['probs'][0]:.2f},{ml['probs'][1]:.2f},{ml['probs'][2]:.2f},{ml['probs'][3]:.2f}]"
            print(f"  {name:30s}  detected={detected_str}  label={label_str}  "
                  f"latency={latency_str}{probs_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
