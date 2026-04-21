"""v0.6 two-stage training: pretrain on real Starlink → finetune on synthetic.

Differences from v0.5 train.py:
  * n_features=12 (6 TLE elements + 6 SGP4 innovation channels)
  * use_physics=False (innovation is now input-feature, physics is frozen
    by being pre-computed offline in preprocess_v06 instead of learned)
  * Two-stage workflow via a single script:
      1. --stage pretrain → self-supervised on data/ml_ready_v06/,
         saves checkpoints/v06_pretrain/best_model.pt
      2. --stage finetune --pretrain-ckpt <path> → supervised on
         data/synthetic_v06/ (via preprocess), saves
         checkpoints/v06_finetune/best_model.pt
  * Model starts medium (6.4M params, roadmap target for v0.6).

v0.6 gate: maneuver accuracy ≥85% AND normal accuracy ≥85% on the
synthetic validation split. Below those numbers we don't promote v0.6
to production and stay on v0.5d+rule_v1+IMM-UKF ensemble.

Usage:
    # Stage 1 — pretrain on real Starlink (227k sequences)
    python -m services.ml.train_v06 --stage pretrain \
        --data data/ml_ready_v06 --epochs 20 \
        --ckpt-out checkpoints/v06_pretrain

    # Stage 2 — finetune on synthetic labels (100k sequences)
    python -m services.ml.train_v06 --stage finetune \
        --data data/ml_ready_v06_synth \
        --pretrain-ckpt checkpoints/v06_pretrain/best_model.pt \
        --epochs 30 --ckpt-out checkpoints/v06_finetune
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
from services.ml.train import FocalLoss, kepler_loss, smoothness_loss

# v0.6 uses 12-feature input: TLE elements + SGP4 innovation
N_FEATURES_V06 = 12


def load_data(data_dir: Path, batch_size: int = 256) -> tuple:
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    assert X_train.shape[-1] == N_FEATURES_V06, (
        f"expected {N_FEATURES_V06} features, got {X_train.shape[-1]}"
    )

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
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  shape: {X_train.shape[1:]}")
    return train_dl, val_dl


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: str,
    pred_weight: float = 1.0,
    cls_weight: float = 1.0,
) -> dict:
    """One training epoch. stage ∈ {pretrain, finetune}."""
    model.train()
    total_pred = 0.0
    total_cls = 0.0
    n_batches = 0

    pred_criterion = nn.MSELoss()
    # Class weights: upweight events over "normal" for finetune
    cls_weights = torch.tensor([0.5, 3.0, 3.0, 3.0], device=device)
    cls_criterion = FocalLoss(weight=cls_weights, gamma=2.0)

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(X, causal=True)
        preds = out["predictions"] if isinstance(out, dict) else out[0]
        cls = out["classifications"] if isinstance(out, dict) else out[1]

        loss = torch.tensor(0.0, device=device)

        # Pretrain: self-supervised next-step prediction on all 12 features
        if stage == "pretrain":
            pred_target = X[:, 1:, :]
            pred_output = preds[:, :-1, :]
            p_loss = pred_criterion(pred_output, pred_target)
            loss = loss + pred_weight * p_loss
            # Physics regularizers only on the 6 TLE-element channels
            loss = loss + 1e-6 * kepler_loss(preds[:, :, :6])
            loss = loss + 1e-4 * smoothness_loss(preds)
            total_pred += p_loss.item()

        # Finetune: supervised classification (primary) + auxiliary pred loss
        if stage == "finetune":
            B, T, C = cls.shape
            c_loss = cls_criterion(cls.reshape(B * T, C), y.reshape(B * T))
            loss = loss + cls_weight * c_loss
            total_cls += c_loss.item()

            # Auxiliary: keep the prediction head aligned with the data
            # during finetune (small weight) so the innovation signal the
            # classifier reads stays meaningful.
            pred_target = X[:, 1:, :]
            pred_output = preds[:, :-1, :]
            p_loss = pred_criterion(pred_output, pred_target)
            loss = loss + 0.1 * p_loss
            total_pred += p_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_batches += 1

    return {
        "pred_loss": total_pred / max(n_batches, 1),
        "cls_loss": total_cls / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stage: str,
) -> dict:
    model.eval()
    total_pred = 0.0
    total_cls = 0.0
    correct = 0
    total = 0
    n_batches = 0
    per_class_correct = torch.zeros(4)
    per_class_total = torch.zeros(4)

    pred_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X, causal=True)
        preds = out["predictions"] if isinstance(out, dict) else out[0]
        cls = out["classifications"] if isinstance(out, dict) else out[1]

        p_loss = pred_criterion(preds[:, :-1, :], X[:, 1:, :])
        total_pred += p_loss.item()

        if stage == "finetune":
            B, T, C = cls.shape
            c_loss = cls_criterion(cls.reshape(B * T, C), y.reshape(B * T))
            total_cls += c_loss.item()

            predicted = cls.argmax(dim=-1)
            correct += (predicted == y).sum().item()
            total += y.numel()
            for k in range(4):
                mask = y == k
                per_class_total[k] += mask.sum().item()
                per_class_correct[k] += ((predicted == y) & mask).sum().item()

        n_batches += 1

    per_class_acc = {
        f"acc_cls{k}": (
            float(per_class_correct[k] / per_class_total[k])
            if per_class_total[k] > 0
            else 0.0
        )
        for k in range(4)
    }
    return {
        "pred_loss": total_pred / max(n_batches, 1),
        "cls_loss": total_cls / max(n_batches, 1),
        "accuracy": correct / max(total, 1) if total > 0 else 0.0,
        **per_class_acc,
    }


def load_pretrain_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device):
    """Load pretrain weights, skipping the classification head (its shape
    is retrainable for finetune). Allows partial-match so changes to the
    cls_head architecture don't break loading."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded pretrain: {len(state)} keys")
    if missing:
        print(f"  missing (expected for finetune head): {len(missing)}")
    if unexpected:
        print(f"  unexpected: {unexpected[:5]}")
    return ckpt.get("epoch", 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrain-ckpt", type=Path, default=None)
    parser.add_argument("--ckpt-out", type=Path, required=True)
    parser.add_argument(
        "--size", choices=["tiny", "small", "medium", "large"], default="medium"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    if args.stage == "finetune" and args.pretrain_ckpt is None:
        print("--pretrain-ckpt required for stage=finetune", file=sys.stderr)
        return 1

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}   Stage: {args.stage}   Size: {args.size}")

    train_dl, val_dl = load_data(args.data, args.batch_size)

    model = create_model(args.size, use_physics=False, n_features=N_FEATURES_V06).to(
        device
    )

    if args.stage == "finetune":
        load_pretrain_checkpoint(model, args.pretrain_ckpt, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.ckpt_out.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    print(
        f"\n{'ep':>3s} {'trP':>9s} {'trC':>9s} {'vaP':>9s} {'vaC':>9s} "
        f"{'acc':>6s} {'c0':>5s} {'c1':>5s} {'c2':>5s} {'c3':>5s} {'sec':>5s}"
    )
    print("-" * 85)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tm = train_epoch(model, train_dl, optimizer, device, args.stage)
        vm = validate(model, val_dl, device, args.stage)
        scheduler.step()
        dt = time.time() - t0
        print(
            f"{ep:>3d} {tm['pred_loss']:>9.5f} {tm['cls_loss']:>9.5f} "
            f"{vm['pred_loss']:>9.5f} {vm['cls_loss']:>9.5f} "
            f"{vm['accuracy']:>6.3f} "
            f"{vm.get('acc_cls0', 0):>5.2f} {vm.get('acc_cls1', 0):>5.2f} "
            f"{vm.get('acc_cls2', 0):>5.2f} {vm.get('acc_cls3', 0):>5.2f} "
            f"{dt:>5.1f}"
        )

        val_loss = vm["pred_loss"] + vm["cls_loss"]
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": vm,
                    "model_config": {
                        "size": args.size,
                        "n_features": N_FEATURES_V06,
                        "use_physics": False,
                        "d_model": model.d_model,
                        "n_classes": model.n_classes,
                    },
                    "stage": args.stage,
                },
                args.ckpt_out / "best_model.pt",
            )

    # v0.6 gate check for finetune
    if args.stage == "finetune":
        final_normal = vm.get("acc_cls0", 0)
        final_maneuver = vm.get("acc_cls1", 0)
        print(f"\nv0.6 GATE:")
        print(
            f"  normal   ≥ 0.85 : {final_normal:.3f} {'PASS' if final_normal >= 0.85 else 'FAIL'}"
        )
        print(
            f"  maneuver ≥ 0.85 : {final_maneuver:.3f} {'PASS' if final_maneuver >= 0.85 else 'FAIL'}"
        )
        if final_normal >= 0.85 and final_maneuver >= 0.85:
            print("  v0.6 GATE PASSED — proceed to Phase 3 OOD validation")
        else:
            print("  v0.6 GATE FAILED — see roadmap Phase 2 fallbacks")

    return 0


if __name__ == "__main__":
    sys.exit(main())
