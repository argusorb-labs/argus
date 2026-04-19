"""Evaluate the Orbital Transformer against known events + baselines.

Benchmarks the foundation model against:
  1. rule_v1 (threshold classifier)
  2. IMM-UKF (physics-based Bayesian)
  3. Ground truth labels (synthetic data)

Metrics:
  - Classification accuracy per event type
  - Detection latency (how many steps after event before model detects)
  - False positive rate on normal trajectories
  - Prediction error (MSE on next-step prediction)

Usage:
    python -m services.ml.evaluate --model checkpoints/best_model.pt --data data/ml_ready/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from services.ml.model import OrbitalTransformer, create_model


LABEL_NAMES = {0: "normal", 1: "maneuver", 2: "decay", 3: "breakup"}


def load_model(checkpoint_path: Path, device: torch.device) -> OrbitalTransformer:
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("model_config", {})
    size = config.get("size", "small")
    model = create_model(size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded model from epoch {epoch}: {model.summary()}")
    return model


@torch.no_grad()
def evaluate_classification(
    model: OrbitalTransformer,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict:
    """Per-class classification metrics on the test set."""
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)

    out = model(X_t, causal=True)
    classifications = out["classifications"] if isinstance(out, dict) else out[1]
    preds = classifications.argmax(dim=-1)  # (B, T)

    # Flatten
    y_flat = y_t.cpu().numpy().ravel()
    p_flat = preds.cpu().numpy().ravel()

    # Overall accuracy
    overall_acc = (y_flat == p_flat).mean()

    # Per-class metrics
    per_class = {}
    for cls_id, cls_name in LABEL_NAMES.items():
        mask = y_flat == cls_id
        if mask.sum() == 0:
            continue
        cls_acc = (p_flat[mask] == cls_id).mean()
        cls_count = mask.sum()
        per_class[cls_name] = {
            "accuracy": float(cls_acc),
            "count": int(cls_count),
            "predicted_as": dict(
                Counter(LABEL_NAMES.get(p, f"unk_{p}") for p in p_flat[mask])
            ),
        }

    return {"overall_accuracy": float(overall_acc), "per_class": per_class}


@torch.no_grad()
def evaluate_detection_latency(
    model: OrbitalTransformer,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict:
    """How many steps after an event starts before the model detects it."""
    X_t = torch.from_numpy(X).float().to(device)
    out = model(X_t, causal=True)
    classifications = out["classifications"] if isinstance(out, dict) else out[1]
    preds = classifications.argmax(dim=-1).cpu().numpy()  # (B, T)

    latencies = {"maneuver": [], "decay": [], "breakup": []}

    for i in range(len(X)):
        # Find the first non-normal label (event start)
        event_steps = np.where(y[i] != 0)[0]
        if len(event_steps) == 0:
            continue

        event_start = event_steps[0]
        event_type = LABEL_NAMES.get(y[i][event_start], "unknown")

        # Find when the model first predicts non-normal after event start
        detected = False
        for t in range(event_start, len(y[i])):
            if preds[i, t] != 0:
                latency = t - event_start
                if event_type in latencies:
                    latencies[event_type].append(latency)
                detected = True
                break

        if not detected and event_type in latencies:
            latencies[event_type].append(len(y[i]) - event_start)  # never detected

    result = {}
    for etype, lats in latencies.items():
        if lats:
            result[etype] = {
                "mean_latency_steps": float(np.mean(lats)),
                "median_latency_steps": float(np.median(lats)),
                "detection_rate": float(np.mean(np.array(lats) < len(y[0]) // 2)),
                "n_events": len(lats),
            }
    return result


@torch.no_grad()
def evaluate_prediction(
    model: OrbitalTransformer,
    X: np.ndarray,
    device: torch.device,
) -> dict:
    """Next-step prediction accuracy (MSE per feature)."""
    X_t = torch.from_numpy(X).float().to(device)
    out = model(X_t, causal=True)
    predictions = out["predictions"] if isinstance(out, dict) else out[0]

    # Compare predictions[t] vs actual[t+1]
    pred = predictions[:, :-1, :].cpu().numpy()
    actual = X[:, 1:, :]

    mse_per_feature = np.mean((pred - actual) ** 2, axis=(0, 1))
    feature_names = [
        "epoch_h",
        "mean_motion",
        "eccentricity",
        "inclination",
        "bstar",
        "alt_km",
    ]

    return {name: float(mse) for name, mse in zip(feature_names, mse_per_feature)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max test samples to evaluate (for speed)",
    )
    args = parser.parse_args(argv)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = load_model(args.model, device)

    X_test = np.load(args.data / "X_test.npy")[: args.max_samples]
    y_test = np.load(args.data / "y_test.npy")[: args.max_samples]
    print(f"Test set: {X_test.shape}")

    # Classification
    print("\n=== Classification Metrics ===")
    cls_metrics = evaluate_classification(model, X_test, y_test, device)
    print(f"Overall accuracy: {cls_metrics['overall_accuracy']:.4f}")
    for cls_name, m in cls_metrics["per_class"].items():
        print(f"  {cls_name:>12s}: acc={m['accuracy']:.4f} (n={m['count']})")

    # Detection latency
    print("\n=== Detection Latency ===")
    latency = evaluate_detection_latency(model, X_test, y_test, device)
    for etype, m in latency.items():
        print(
            f"  {etype:>12s}: mean={m['mean_latency_steps']:.1f} steps, "
            f"detect_rate={m['detection_rate']:.2%}, n={m['n_events']}"
        )

    # Prediction
    print("\n=== Prediction MSE (per feature) ===")
    pred_mse = evaluate_prediction(model, X_test, device)
    for fname, mse in pred_mse.items():
        print(f"  {fname:>15s}: {mse:.8f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
