"""Training script for the Orbital Transformer.

Supports two training modes:
  1. Self-supervised (predict next TLE): uses prediction head loss
  2. Supervised (classify events): uses classification head loss
  3. Mixed (both): weighted combination

Usage:
    # Train on synthetic data (supervised)
    python -m services.ml.train --data data/ml_ready/ --mode supervised --epochs 50

    # Pre-train on Space-Track (self-supervised)
    python -m services.ml.train --data data/ml_ready/ --mode selfsup --epochs 20

    # Mixed training
    python -m services.ml.train --data data/ml_ready/ --mode mixed --epochs 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from services.ml.model import create_model

# ── Physics-constrained loss terms ──


def kepler_loss(predictions: torch.Tensor) -> torch.Tensor:
    """Penalize violations of Kepler's third law: mm² × a³ ≈ const.

    In normalized space, mean_motion is feature index 1, alt_km is index 5.
    We check that (mm_pred)² × (R + alt_pred)³ is approximately constant
    across timesteps.
    """
    mm = predictions[:, :, 1]  # normalized mean motion
    alt = predictions[:, :, 5]  # normalized altitude

    # Kepler invariant (in normalized space, relative change matters)
    kepler_val = mm**2 * (alt + 1) ** 3  # +1 to avoid zero
    kepler_diff = kepler_val[:, 1:] - kepler_val[:, :-1]
    return torch.mean(kepler_diff**2)


def smoothness_loss(predictions: torch.Tensor) -> torch.Tensor:
    """Penalize non-smooth predictions (second-order finite difference)."""
    if predictions.size(1) < 3:
        return torch.tensor(0.0, device=predictions.device)
    d2 = predictions[:, 2:, :] - 2 * predictions[:, 1:-1, :] + predictions[:, :-2, :]
    return torch.mean(d2**2)


class MemmapDataset(torch.utils.data.Dataset):
    """Dataset backed by memory-mapped numpy files (no full RAM load)."""

    def __init__(self, X_path: Path, y_path: Path, max_samples: int = 0):
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        self.n = min(len(self.X), max_samples) if max_samples > 0 else len(self.X)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx].copy()).float(),
            torch.from_numpy(self.y[idx].copy()).long(),
        )


def load_data(data_dir: Path, batch_size: int = 256, max_train: int = 0) -> tuple:
    """Load preprocessed train/val numpy arrays into DataLoaders.

    Uses memory-mapped files so large datasets don't need full RAM.
    """
    train_ds = MemmapDataset(
        data_dir / "X_train.npy", data_dir / "y_train.npy", max_samples=max_train
    )
    val_ds = MemmapDataset(
        data_dir / "X_val.npy", data_dir / "y_val.npy",
        max_samples=max_train // 8 if max_train > 0 else 0,
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    print(f"Train: {len(train_ds):,} sequences, Val: {len(val_ds):,} sequences")
    print(f"Sequence shape: {train_ds.X.shape[1:]} (steps × features)")
    return train_dl, val_dl


class FocalLoss(nn.Module):
    """Focal Loss with class weights.

    Combines two ideas:
    - Class weights: "maneuver is important" (fixed per-class multiplier)
    - Focal modulation: "this sample is hard" (dynamic per-sample multiplier)

    FL = -w_y · (1 - p_y)^γ · log(p_y)

    When γ=0, this is exactly weighted CrossEntropy.
    When γ>0, easy samples (high p_y) get loss → 0, hard samples dominate.
    γ=2 is the standard value from the RetinaNet paper (Lin et al., 2017).
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        p = torch.softmax(logits, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * ce).mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str = "mixed",
    pred_weight: float = 1.0,
    cls_weight: float = 1.0,
) -> dict:
    """Train one epoch. Returns loss dict."""
    model.train()
    total_pred_loss = 0.0
    total_cls_loss = 0.0
    n_batches = 0

    pred_criterion = nn.MSELoss()
    # Focal Loss with class weights: combines "this class is important"
    # (weights) with "this sample is hard" (focal γ modulation).
    cls_weights = torch.tensor([0.5, 3.0, 3.0, 3.0], device=device)
    cls_criterion = FocalLoss(weight=cls_weights, gamma=2.0)

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(X, causal=True)
        # v0.5: model returns dict with predictions, classifications, etc.
        if isinstance(out, dict):
            predictions = out["predictions"]
            classifications = out["classifications"]
        else:
            predictions, classifications = out

        loss = torch.tensor(0.0, device=device)

        # Prediction loss: predict next step from current
        if mode in ("selfsup", "mixed"):
            pred_target = X[:, 1:, :]
            pred_output = predictions[:, :-1, :]
            p_loss = pred_criterion(pred_output, pred_target)
            loss = loss + pred_weight * p_loss
            total_pred_loss += p_loss.item()

            # Physics constraints — weight must be tiny relative to data loss.
            # Kepler/smoothness values are in normalized space where the
            # numbers can be large. 1e-6 keeps them as gentle regularization.
            loss = loss + 1e-6 * kepler_loss(predictions)
            loss = loss + 1e-4 * smoothness_loss(predictions)

        # Classification loss
        if mode in ("supervised", "mixed"):
            B, T, C = classifications.shape
            c_loss = cls_criterion(
                classifications.reshape(B * T, C),
                y.reshape(B * T),
            )
            loss = loss + cls_weight * c_loss
            total_cls_loss += c_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_batches += 1

    return {
        "pred_loss": total_pred_loss / max(n_batches, 1),
        "cls_loss": total_cls_loss / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str = "mixed",
) -> dict:
    """Validate. Returns loss + accuracy dict."""
    model.eval()
    total_pred_loss = 0.0
    total_cls_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    pred_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X, causal=True)
        if isinstance(out, dict):
            predictions = out["predictions"]
            classifications = out["classifications"]
        else:
            predictions, classifications = out

        if mode in ("selfsup", "mixed"):
            p_loss = pred_criterion(predictions[:, :-1, :], X[:, 1:, :])
            total_pred_loss += p_loss.item()

        if mode in ("supervised", "mixed"):
            B, T, C = classifications.shape
            c_loss = cls_criterion(
                classifications.reshape(B * T, C),
                y.reshape(B * T),
            )
            total_cls_loss += c_loss.item()

            preds = classifications.argmax(dim=-1)  # (B, T)
            correct += (preds == y).sum().item()
            total += y.numel()

        n_batches += 1

    acc = correct / max(total, 1) if total > 0 else 0.0
    return {
        "pred_loss": total_pred_loss / max(n_batches, 1),
        "cls_loss": total_cls_loss / max(n_batches, 1),
        "accuracy": acc,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the Orbital Transformer.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory with X_train.npy, y_train.npy, etc.",
    )
    parser.add_argument(
        "--mode", choices=["selfsup", "supervised", "mixed"], default="mixed"
    )
    parser.add_argument(
        "--size", choices=["tiny", "small", "medium", "large"], default="small"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--max-train", type=int, default=0,
                        help="Limit training samples (0 = all)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint (e.g. checkpoints/v07b/best_model.pt)")
    args = parser.parse_args(argv)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    train_dl, val_dl = load_data(args.data, args.batch_size, max_train=args.max_train)

    # Model
    model = create_model(args.size).to(device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded epoch {ckpt.get('epoch', '?')} "
              f"(val_acc={ckpt.get('val_metrics', {}).get('accuracy', '?')})")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nTraining: {args.epochs} epochs, mode={args.mode}, lr={args.lr}")
    print(
        f"{'epoch':>5s}  {'train_pred':>10s}  {'train_cls':>10s}  "
        f"{'val_pred':>10s}  {'val_cls':>10s}  {'val_acc':>8s}  {'time':>6s}  {'lr':>10s}"
    )
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_dl, optimizer, device, mode=args.mode)
        val_metrics = validate(model, val_dl, device, mode=args.mode)

        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:>5d}  {train_metrics['pred_loss']:>10.6f}  "
            f"{train_metrics['cls_loss']:>10.6f}  "
            f"{val_metrics['pred_loss']:>10.6f}  "
            f"{val_metrics['cls_loss']:>10.6f}  "
            f"{val_metrics['accuracy']:>8.4f}  "
            f"{elapsed:>5.1f}s  {lr:>10.2e}"
        )

        # Checkpoint best model
        val_loss = val_metrics["pred_loss"] + val_metrics["cls_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = args.checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "model_config": {
                        "size": args.size,
                        "n_features": model.n_features,
                        "d_model": model.d_model,
                        "n_classes": model.n_classes,
                    },
                },
                ckpt_path,
            )

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {args.checkpoint_dir}/best_model.pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
