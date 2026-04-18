"""Interactive Multiple Model (IMM) estimator.

Runs N UKF instances in parallel, each with a different dynamics model
(different process noise Q). At each step, the model probabilities
are updated via Bayesian model comparison using each filter's
log-likelihood.

For ArgusOrb SSA:
  M1 (station-keeping): small Q — elements change slowly
  M2 (maneuver):        large Q — allows sudden Δv
  M3 (decay):           large Q on along-track, drift term

The model posterior P(Mk|data) IS the anomaly label confidence.
When P(M2) spikes, the satellite likely maneuvered. When P(M3)
dominates, it's decaying.

Reference: Blom & Bar-Shalom (1988), "The Interacting Multiple Model
Algorithm for Systems with Markovian Switching Coefficients."
"""

from __future__ import annotations

import numpy as np

from services.brain.ukf import UKF


class IMM:
    """Interactive Multiple Model estimator wrapping N UKF filters."""

    def __init__(
        self,
        filters: list[UKF],
        model_probs: np.ndarray | None = None,
        transition_matrix: np.ndarray | None = None,
    ):
        """
        Args:
            filters: list of N UKF instances, each configured with its own Q
            model_probs: initial model probabilities (N,), uniform if None
            transition_matrix: Markov transition matrix (N, N) — element [i,j]
                is P(switch from model i to model j). Should be row-stochastic.
                Default: high self-transition (0.98), low switch probability.
        """
        self.filters = filters
        self.n_models = len(filters)
        n = self.n_models

        if model_probs is not None:
            self.mu = np.array(model_probs, dtype=float)
        else:
            self.mu = np.ones(n) / n

        if transition_matrix is not None:
            self.T = np.array(transition_matrix, dtype=float)
        else:
            # Default: 98% stay in current model, 1% switch to each other
            self.T = np.full((n, n), 0.01)
            np.fill_diagonal(self.T, 1.0 - 0.01 * (n - 1))

        # Combined state estimate (updated after each cycle)
        self.x = filters[0].x.copy()
        self.P = filters[0].P.copy()

    def predict(self, dt: float, fx_args_per_model: list[tuple] | None = None) -> None:
        """IMM predict: mix → predict each filter.

        Args:
            dt: time step in seconds
            fx_args_per_model: optional per-model args for fx. If None,
                all models get empty args. Example: [(bstar,), (bstar,), (bstar,)]
        """
        n = self.n_models

        if fx_args_per_model is None:
            fx_args_per_model = [() for _ in range(n)]

        # ── Step 1: Compute mixing probabilities ──
        # c_bar[j] = sum_i T[i,j] * mu[i]
        c_bar = self.T.T @ self.mu  # (n,)
        c_bar = np.maximum(c_bar, 1e-30)  # avoid division by zero

        # mu_ij[i,j] = P(model i at t-1 | model j at t) = T[i,j] * mu[i] / c_bar[j]
        mu_ij = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mu_ij[i, j] = self.T[i, j] * self.mu[i] / c_bar[j]

        # ── Step 2: Mix state estimates ──
        for j in range(n):
            x_mixed = np.zeros_like(self.filters[j].x)
            for i in range(n):
                x_mixed += mu_ij[i, j] * self.filters[i].x
            P_mixed = np.zeros_like(self.filters[j].P)
            for i in range(n):
                dx = self.filters[i].x - x_mixed
                P_mixed += mu_ij[i, j] * (self.filters[i].P + np.outer(dx, dx))
            self.filters[j].x = x_mixed
            self.filters[j].P = P_mixed

        # ── Step 3: Predict each filter ──
        for j in range(n):
            self.filters[j].predict(dt=dt, fx_args=fx_args_per_model[j])

        # Update mixing probabilities (before update step)
        self.mu = c_bar

    def update(self, z: np.ndarray) -> None:
        """IMM update: update each filter, then combine."""
        n = self.n_models

        # ── Step 4: Update each filter ──
        likelihoods = np.zeros(n)
        for j in range(n):
            self.filters[j].update(z)
            likelihoods[j] = np.exp(self.filters[j].log_likelihood)

        # ── Step 5: Update model probabilities ──
        c = self.mu * likelihoods
        c_sum = np.sum(c)
        if c_sum > 0:
            self.mu = c / c_sum
        else:
            # All likelihoods near zero — reset to uniform
            self.mu = np.ones(n) / n

        # ── Step 6: Combine state estimates ──
        self.x = np.zeros_like(self.filters[0].x)
        for j in range(n):
            self.x += self.mu[j] * self.filters[j].x

        self.P = np.zeros_like(self.filters[0].P)
        for j in range(n):
            dx = self.filters[j].x - self.x
            self.P += self.mu[j] * (self.filters[j].P + np.outer(dx, dx))

    @property
    def model_probabilities(self) -> dict[int, float]:
        """Model index → posterior probability."""
        return {i: float(self.mu[i]) for i in range(self.n_models)}

    @property
    def most_likely_model(self) -> int:
        """Index of the most probable model."""
        return int(np.argmax(self.mu))
