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


def load_data(data_dir: Path, batch_size: int = 256) -> tuple:
    """Load preprocessed train/val numpy arrays into DataLoaders."""
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_ds)} sequences, Val: {len(val_ds)} sequences")
    print(f"Sequence shape: {X_train.shape[1:]} (steps × features)")
    return train_dl, val_dl


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
    cls_criterion = nn.CrossEntropyLoss()

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        predictions, classifications = model(X, causal=True)

        loss = torch.tensor(0.0, device=device)

        # Prediction loss: predict next step from current
        if mode in ("selfsup", "mixed"):
            # Shift: predict X[t+1] from context up to X[t]
            pred_target = X[:, 1:, :]  # (B, T-1, F)
            pred_output = predictions[:, :-1, :]  # (B, T-1, F)
            p_loss = pred_criterion(pred_output, pred_target)
            loss = loss + pred_weight * p_loss
            total_pred_loss += p_loss.item()

        # Classification loss
        if mode in ("supervised", "mixed"):
            # Reshape for CrossEntropyLoss: (B*T, C) vs (B*T,)
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
        predictions, classifications = model(X, causal=True)

        if mode in ("selfsup", "mixed"):
            p_loss = pred_criterion(predictions[:, :-1, :], X[:, 1:, :])
            total_pred_loss += p_loss.item()

        if mode in ("supervised", "mixed"):
            B, T, C = classifications.shape
            c_loss = cls_criterion(
                classifications.reshape(B * T, C), y.reshape(B * T),
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
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory with X_train.npy, y_train.npy, etc.")
    parser.add_argument("--mode", choices=["selfsup", "supervised", "mixed"],
                        default="mixed")
    parser.add_argument("--size", choices=["tiny", "small", "medium", "large"],
                        default="small")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="auto",
                        help="cpu, cuda, mps, or auto")
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
    train_dl, val_dl = load_data(args.data, args.batch_size)

    # Model
    model = create_model(args.size).to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nTraining: {args.epochs} epochs, mode={args.mode}, lr={args.lr}")
    print(f"{'epoch':>5s}  {'train_pred':>10s}  {'train_cls':>10s}  "
          f"{'val_pred':>10s}  {'val_cls':>10s}  {'val_acc':>8s}  {'time':>6s}  {'lr':>10s}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_dl, optimizer, device, mode=args.mode
        )
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
            torch.save({
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
            }, ckpt_path)

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {args.checkpoint_dir}/best_model.pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
