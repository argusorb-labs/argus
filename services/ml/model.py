"""Orbital Transformer — physics-informed foundation model for SSA.

v0.5 architecture: physics-residual + dual-head.

  output = f_physics(x) + f_neural(x)
                            ↓
           prediction_head (next-step regression)
           noise_head      (predicted noise σ per timestep)
           anomaly_head    (per-step event classification)

  anomaly_score = ||f_neural|| / noise_head = signal-to-noise ratio

f_physics implements known orbital mechanics (J2 secular drift, drag
decay, eccentricity circularization). f_neural (the transformer) learns
ONLY what physics can't explain. The anomaly score is the SNR — how
much of the residual exceeds expected noise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Physics constants for element-space prediction ──
# These implement simplified analytical propagation in TLE element space.

MU_EARTH = 3.986004418e14  # m³/s²
R_EARTH_KM = 6371.0


class AnalyticalPhysics(nn.Module):
    """Differentiable analytical physics: predict next elements from current.

    Implements in TLE element space (not Cartesian):
    - mean_motion increases due to drag (proportional to B*)
    - eccentricity decreases due to drag (circularization)
    - inclination approximately conserved (no thrust)
    - B* slowly varying
    - altitude derived from mean_motion

    These are crude but capture the first-order dynamics. The neural
    residual f_neural corrects everything this gets wrong.
    """

    def __init__(self):
        super().__init__()
        # Learnable scaling factors (initialized to empirical values,
        # fine-tuned during training so physics adapts to the data)
        self.drag_mm_scale = nn.Parameter(torch.tensor(0.001))
        self.drag_ecc_scale = nn.Parameter(torch.tensor(0.0001))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next timestep's elements from current.

        Args:
            x: (batch, seq_len, 6) — [epoch_h, mm, ecc, incl, bstar, alt_km]
               (normalized)

        Returns:
            (batch, seq_len, 6) — predicted next elements
        """
        # Physics prediction: each element at t predicts element at t+1
        # We shift by one position to align with "next step"
        epoch = x[:, :, 0]
        mm = x[:, :, 1]
        ecc = x[:, :, 2]
        incl = x[:, :, 3]
        bstar = x[:, :, 4]
        alt = x[:, :, 5]

        # Drag effect: mean motion increases (orbit shrinks)
        # In normalized space, the change is scaled by bstar
        delta_mm = self.drag_mm_scale * bstar
        pred_mm = mm + delta_mm

        # Drag circularizes: eccentricity decreases
        delta_ecc = -self.drag_ecc_scale * torch.abs(bstar)
        pred_ecc = ecc + delta_ecc

        # Inclination: conserved (no thrust)
        pred_incl = incl

        # B*: slowly varying, predict same
        pred_bstar = bstar

        # Altitude: inversely related to mean motion change
        pred_alt = alt - delta_mm * 10  # empirical scaling

        # Epoch: advances by step
        pred_epoch = epoch

        return torch.stack([
            pred_epoch, pred_mm, pred_ecc, pred_incl, pred_bstar, pred_alt
        ], dim=-1)


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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class OrbitalTransformer(nn.Module):
    """Physics-informed temporal transformer for orbital anomaly detection.

    v0.5: physics-residual architecture with three output heads.

    Args:
        n_features: input feature dimension (default 6)
        d_model: embedding dimension (default 128)
        n_heads: attention heads (default 4)
        n_layers: transformer encoder layers (default 4)
        n_classes: event classification classes (default 4)
        dropout: dropout rate (default 0.1)
        max_len: maximum sequence length (default 500)
        use_physics: enable physics-residual branch (default True)
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
        use_physics: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_classes = n_classes
        self.use_physics = use_physics

        # Physics branch
        if use_physics:
            self.physics = AnalyticalPhysics()

        # Neural branch: input projection
        self.input_proj = nn.Linear(n_features, d_model)
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

        # Head 1: Prediction (next-step regression)
        # Outputs the RESIDUAL on top of physics (if use_physics)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

        # Head 2: Noise estimation (predicted σ per timestep)
        # Learns "how noisy should this observation be given the context"
        self.noise_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # σ must be positive
        )

        # Head 3: Classification (per-step event type)
        # Fed by concatenation of: transformer hidden state h (rich context)
        # + f_neural (physics residual) + anomaly_score (SNR signal).
        # This prevents the "starvation" problem where f_neural ≈ 0
        # (physics is accurate) starves the classifier of gradient.
        cls_input_dim = d_model + n_features + 1  # h + f_neural + anomaly_score
        self.cls_head = nn.Sequential(
            nn.Linear(cls_input_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

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
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features) — normalized orbital elements
            causal: if True, use causal mask

        Returns dict with:
            predictions: (B, T, n_features) — predicted next step
            f_neural: (B, T, n_features) — neural residual (before adding physics)
            noise_sigma: (B, T, 1) — predicted noise σ
            classifications: (B, T, n_classes) — event logits
            anomaly_score: (B, T, 1) — ||f_neural|| / noise_sigma
        """
        B, T, F = x.shape

        # Neural branch: transformer
        h = self.input_proj(x)
        h = self.pos_enc(h)

        if causal:
            mask = self._get_causal_mask(T, x.device)
            h = self.encoder(h, mask=mask)
        else:
            h = self.encoder(h)

        # Neural residual
        f_neural = self.pred_head(h)  # (B, T, F)

        # Physics prediction
        if self.use_physics:
            f_physics = self.physics(x)  # (B, T, F)
            predictions = f_physics + f_neural
        else:
            predictions = f_neural

        # Noise estimation
        noise_sigma = self.noise_head(h)  # (B, T, 1)
        noise_sigma = noise_sigma.clamp(min=1e-6)

        # Anomaly score = SNR
        f_neural_norm = torch.norm(f_neural, dim=-1, keepdim=True)  # (B, T, 1)
        anomaly_score = f_neural_norm / noise_sigma  # (B, T, 1)

        # Classification: feed h + f_neural + anomaly_score
        cls_input = torch.cat([h, f_neural, anomaly_score], dim=-1)
        classifications = self.cls_head(cls_input)  # (B, T, n_classes)

        return {
            "predictions": predictions,
            "f_neural": f_neural,
            "f_physics": f_physics if self.use_physics else torch.zeros_like(f_neural),
            "noise_sigma": noise_sigma,
            "anomaly_score": anomaly_score,
            "classifications": classifications,
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        n = self.num_parameters
        if n < 1e6:
            size = f"{n / 1e3:.0f}K"
        else:
            size = f"{n / 1e6:.1f}M"
        physics = "+physics" if self.use_physics else ""
        return (
            f"OrbitalTransformer{physics}: {size} params, "
            f"d_model={self.d_model}, layers={len(self.encoder.layers)}, "
            f"features={self.n_features}, classes={self.n_classes}"
        )


def create_model(size: str = "small", use_physics: bool = True) -> OrbitalTransformer:
    """Factory for pre-configured model sizes."""
    configs = {
        "tiny": dict(d_model=64, n_heads=2, n_layers=2),
        "small": dict(d_model=128, n_heads=4, n_layers=4),
        "medium": dict(d_model=256, n_heads=8, n_layers=8),
        "large": dict(d_model=512, n_heads=8, n_layers=12),
    }
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs)}")

    model = OrbitalTransformer(**configs[size], use_physics=use_physics)
    print(f"Created model: {model.summary()}")
    return model
