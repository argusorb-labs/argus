"""Orbital Transformer — foundation model for SSA anomaly detection.

A temporal transformer that processes sequences of orbital elements
and produces two outputs:
  1. Prediction head: predict the next timestep (self-supervised)
  2. Classification head: classify each timestep (supervised)

The model is TINY by LLM standards (1-30M params vs GPT-2's 124M).
Training on a single GPU takes hours, not weeks.

Architecture:
  Input embedding → Positional encoding → N × TransformerEncoder
  → Prediction head (next-step regression)
  → Classification head (per-step event type)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class OrbitalTransformer(nn.Module):
    """Temporal transformer for orbital element sequences.

    Args:
        n_features: input feature dimension (default 6)
        d_model: embedding dimension (default 128)
        n_heads: attention heads (default 4)
        n_layers: transformer encoder layers (default 4)
        n_classes: event classification classes (default 4)
        dropout: dropout rate (default 0.1)
        max_len: maximum sequence length (default 500)
    """

    def __init__(
        self,
        n_features: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_classes: int = 4,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head: predict next timestep's features
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

        # Classification head: per-timestep event classification
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        # Causal mask for autoregressive prediction
        self._causal_mask_cache: dict[int, torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len not in self._causal_mask_cache:
            mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]

    def forward(
        self, x: torch.Tensor, causal: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features) — normalized orbital elements
            causal: if True, use causal mask (for autoregressive training)

        Returns:
            predictions: (batch, seq_len, n_features) — predicted next step
            classifications: (batch, seq_len, n_classes) — event logits
        """
        # Embed
        h = self.input_proj(x)  # (B, T, d_model)
        h = self.pos_enc(h)

        # Transformer encode
        if causal:
            mask = self._get_causal_mask(x.size(1), x.device)
            h = self.encoder(h, mask=mask)
        else:
            h = self.encoder(h)

        # Heads
        predictions = self.pred_head(h)      # (B, T, n_features)
        classifications = self.cls_head(h)   # (B, T, n_classes)

        return predictions, classifications

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        n = self.num_parameters
        if n < 1e6:
            size = f"{n/1e3:.0f}K"
        else:
            size = f"{n/1e6:.1f}M"
        return (
            f"OrbitalTransformer: {size} params, "
            f"d_model={self.d_model}, layers={len(self.encoder.layers)}, "
            f"features={self.n_features}, classes={self.n_classes}"
        )


def create_model(size: str = "small") -> OrbitalTransformer:
    """Factory for pre-configured model sizes.

    Sizes:
        tiny:  ~200K params (fast prototyping)
        small: ~1M params (default prototype)
        medium: ~10M params (serious training)
        large: ~30M params (production)
    """
    configs = {
        "tiny": dict(d_model=64, n_heads=2, n_layers=2),
        "small": dict(d_model=128, n_heads=4, n_layers=4),
        "medium": dict(d_model=256, n_heads=8, n_layers=8),
        "large": dict(d_model=512, n_heads=8, n_layers=12),
    }
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs)}")

    model = OrbitalTransformer(**configs[size])
    print(f"Created model: {model.summary()}")
    return model
