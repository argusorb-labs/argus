"""ML inference service for production anomaly detection.

Loads a trained OrbitalTransformer checkpoint and classifies TLE sequences.
Designed to run alongside rule_v1 and IMM-UKF in the fetch cycle, writing
labels with classified_by="ml_v1" to the anomaly table.

Usage in production:
    classifier = MLClassifier("checkpoints/v11/best_model.pt")
    labels = classifier.classify_satellites(store, norad_ids)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from services.ml.model import create_model

# Same normalization as training pipeline (label_spacetrack_bulk.py)
FEATURE_MEANS = np.array([660.41, 12.169, 0.0616, 74.028, 0.0, 4224.95, 22.936,
                           179.45, 174.79, 184.01, 0.0001], dtype=np.float32)
FEATURE_STDS = np.array([2170.74, 4.256, 0.1681, 27.209, 0.01, 8856.03, 26.795,
                          104.95, 103.39, 107.97, 0.0016], dtype=np.float32)
FEATURE_STDS = np.maximum(FEATURE_STDS, 1e-10)

LABEL_NAMES = {0: "normal", 1: "maneuver", 2: "decay", 3: "breakup"}
SEQ_LEN = 50


class MLClassifier:
    """Production ML classifier wrapping OrbitalTransformer."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        config = ckpt.get("model_config", {})
        size = config.get("size", "medium")
        n_features = config.get("n_features", 11)

        self.model = create_model(size, n_features=n_features)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.n_features = n_features

        params = sum(p.numel() for p in self.model.parameters())
        print(f"[ML] Loaded {size} model ({params:,} params, {n_features}D) "
              f"from {checkpoint_path}")

    def _tle_records_to_features(self, records: list[dict]) -> np.ndarray | None:
        """Convert TLE records to a (1, T, 11) normalized feature array.

        Records must be sorted by epoch (oldest first) and have at least
        SEQ_LEN entries. Takes the last SEQ_LEN records.
        """
        if len(records) < SEQ_LEN:
            return None

        records = records[-SEQ_LEN:]
        arr = np.zeros((SEQ_LEN, 11), dtype=np.float64)

        def _epoch_hours(r):
            year = r.get("epoch_year", 0)
            day = r.get("epoch_day", 0)
            if not year:
                # Try epoch_jd
                jd = r.get("epoch_jd", 0)
                if jd:
                    return jd * 24.0
                return 0.0
            return (year * 365.25 + day) * 24.0

        epoch_h = np.array([_epoch_hours(r) for r in records])

        for j, r in enumerate(records):
            arr[j, 0] = epoch_h[j]
            arr[j, 1] = r.get("mean_motion", 0) or 0
            arr[j, 2] = r.get("eccentricity", 0) or 0
            arr[j, 3] = r.get("inclination", 0) or 0
            arr[j, 4] = r.get("bstar", 0) or 0
            arr[j, 5] = r.get("alt_km", 0) or 0
            arr[j, 7] = r.get("raan", 0) or 0
            arr[j, 8] = r.get("argp", 0) or 0
            arr[j, 9] = r.get("mean_anomaly", 0) or 0
            arr[j, 10] = r.get("n_dot", 0) or 0

        # Relative epoch
        arr[:, 0] -= arr[0, 0]

        # dt_hours
        arr[0, 6] = 0.0
        if len(epoch_h) > 1:
            arr[1:, 6] = np.diff(epoch_h)
        arr[:, 6] = np.clip(arr[:, 6], 0.0, 240.0)

        # Clip B* and normalize
        arr[:, 4] = np.clip(arr[:, 4], -1.0, 1.0)

        # Use only the features the model expects
        arr = arr[:, :self.n_features]
        means = FEATURE_MEANS[:self.n_features]
        stds = FEATURE_STDS[:self.n_features]

        arr_norm = ((arr - means) / stds).astype(np.float32)
        return arr_norm[np.newaxis, :, :]  # (1, T, F)

    @torch.no_grad()
    def classify_sequence(self, X: np.ndarray) -> list[dict]:
        """Classify a single normalized sequence.

        Args:
            X: (1, T, F) normalized feature array

        Returns:
            list of dicts with classification per timestep
        """
        x = torch.from_numpy(X).float().to(self.device)
        out = self.model(x, causal=True)
        cls = out["classifications"][0]  # (T, 4)
        probs = torch.softmax(cls, dim=-1).cpu().numpy()
        preds = cls.argmax(dim=-1).cpu().numpy()
        return [
            {"label": int(preds[t]), "probs": probs[t].tolist()}
            for t in range(len(preds))
        ]

    def classify_satellite(
        self, store, norad_id: int, max_history: int = 100,
    ) -> list[dict]:
        """Classify a satellite's recent TLE history.

        Returns anomaly labels (non-normal only) ready for store.insert_anomaly().
        """
        history = store.get_satellite_history(norad_id, limit=max_history)
        if len(history) < SEQ_LEN:
            return []

        # Sort oldest first
        history.sort(key=lambda r: r.get("epoch_jd", 0))

        X = self._tle_records_to_features(history)
        if X is None:
            return []

        results = self.classify_sequence(X)
        labels = []

        # Only emit labels for the last few timesteps (most recent TLEs)
        # to avoid re-labeling old data every cycle
        for t in range(max(0, len(results) - 5), len(results)):
            r = results[t]
            if r["label"] == 0:
                continue

            tle_record = history[-(SEQ_LEN - t)] if t < SEQ_LEN else history[-1]
            label_name = LABEL_NAMES[r["label"]]
            probs = r["probs"]

            labels.append({
                "norad_id": norad_id,
                "anomaly_type": f"ml_{label_name}",
                "cause": label_name,
                "confidence": round(float(max(probs)), 4),
                "classified_by": "ml_v1",
                "source_epoch_jd": tle_record.get("epoch_jd"),
                "detected_at": time.time(),
                "altitude_before_km": None,
                "altitude_after_km": round(tle_record.get("alt_km", 0) or 0, 1),
                "details": (
                    f"ML: P(norm)={probs[0]:.3f} P(man)={probs[1]:.3f} "
                    f"P(dec)={probs[2]:.3f} P(brk)={probs[3]:.3f}"
                ),
            })

        return labels

    def classify_satellites(
        self, store, norad_ids: list[int], max_history: int = 100,
    ) -> list[dict]:
        """Classify multiple satellites. Returns all anomaly labels."""
        all_labels = []
        for nid in norad_ids:
            try:
                labels = self.classify_satellite(store, nid, max_history)
                all_labels.extend(labels)
            except Exception:
                continue
        return all_labels
