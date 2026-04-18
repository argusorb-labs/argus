"""Unscented Kalman Filter — sigma-point state estimation.

Implements the UKF with the Julier-Uhlmann sigma point selection.
No linearization needed (unlike EKF) — sigma points are propagated
through the nonlinear dynamics directly, which handles orbital
mechanics well.

This is a generic UKF implementation. The orbital-specific parts
(state transition via dynamics.propagate_state, observation model)
are injected via callables.

Reference: Wan & van der Merwe (2000), "The Unscented Kalman Filter
for Nonlinear Estimation."
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _sigma_points(x: np.ndarray, P: np.ndarray, alpha: float = 1e-3,
                  beta: float = 2.0, kappa: float = 0.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate 2n+1 sigma points and their weights.

    Args:
        x: state mean (n,)
        P: state covariance (n, n)
        alpha: spread parameter (1e-3 typical)
        beta: distribution parameter (2 = Gaussian)
        kappa: secondary scaling (0 or 3-n)

    Returns:
        (sigma_points, weights_mean, weights_cov)
        sigma_points: (2n+1, n)
        weights_mean: (2n+1,)
        weights_cov: (2n+1,)
    """
    n = len(x)
    lam = alpha**2 * (n + kappa) - n

    # Square root of (n + lambda) * P via Cholesky
    try:
        S = np.linalg.cholesky((n + lam) * P)
    except np.linalg.LinAlgError:
        # P not positive definite — add small diagonal and retry
        S = np.linalg.cholesky((n + lam) * (P + 1e-10 * np.eye(n)))

    sigmas = np.zeros((2 * n + 1, n))
    sigmas[0] = x
    for i in range(n):
        sigmas[i + 1] = x + S[i]
        sigmas[n + i + 1] = x - S[i]

    # Weights
    wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    return sigmas, wm, wc


class UKF:
    """Unscented Kalman Filter.

    Usage:
        ukf = UKF(n_state=6, n_obs=6, fx=propagate, hx=observe)
        ukf.x = initial_state
        ukf.P = initial_covariance
        ukf.Q = process_noise
        ukf.R = observation_noise

        ukf.predict(dt=dt, fx_args=(bstar,))
        ukf.update(z=observation)
    """

    def __init__(
        self,
        n_state: int,
        n_obs: int,
        fx: Callable,
        hx: Callable,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """
        Args:
            n_state: state dimension
            n_obs: observation dimension
            fx: state transition function f(sigma, dt, *args) -> sigma_next
            hx: observation function h(sigma) -> z_predicted
            alpha, beta, kappa: sigma point parameters
        """
        self.n_state = n_state
        self.n_obs = n_obs
        self.fx = fx
        self.hx = hx
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # State and covariance
        self.x = np.zeros(n_state)
        self.P = np.eye(n_state)

        # Noise covariances
        self.Q = np.eye(n_state) * 1e-6    # process noise
        self.R = np.eye(n_obs) * 1e-4       # observation noise

        # Innovation (for diagnostics / IMM)
        self.innovation = np.zeros(n_obs)
        self.innovation_cov = np.eye(n_obs)
        self.log_likelihood = 0.0

    def predict(self, dt: float, fx_args: tuple = (),
                batch_fx: callable | None = None) -> None:
        """Predict step: propagate sigma points through dynamics.

        Args:
            dt: time step
            fx_args: extra args passed to self.fx
            batch_fx: optional batch propagation function
                f(sigmas_array, dt, *args) -> sigmas_pred_array.
                If provided, all sigma points are propagated in one call
                (much faster than 13 sequential calls).
        """
        sigmas, wm, wc = _sigma_points(
            self.x, self.P, self.alpha, self.beta, self.kappa
        )
        n_sigma = len(sigmas)
        n = self.n_state

        # Propagate sigma points
        if batch_fx is not None:
            sigmas_pred, _ = batch_fx(sigmas, dt, *fx_args)
        else:
            sigmas_pred = np.zeros_like(sigmas)
            for i in range(n_sigma):
                sigmas_pred[i] = self.fx(sigmas[i], dt, *fx_args)

        # Predicted mean
        x_pred = np.sum(wm[:, None] * sigmas_pred, axis=0)

        # Predicted covariance
        P_pred = np.zeros((n, n))
        for i in range(n_sigma):
            d = sigmas_pred[i] - x_pred
            P_pred += wc[i] * np.outer(d, d)
        P_pred += self.Q

        self.x = x_pred
        self.P = P_pred
        self._sigmas_pred = sigmas_pred
        self._wm = wm
        self._wc = wc

    def update(self, z: np.ndarray) -> None:
        """Update step: incorporate observation z."""
        sigmas_pred = self._sigmas_pred
        wm = self._wm
        wc = self._wc
        n_sigma = len(sigmas_pred)
        n = self.n_state
        m = self.n_obs

        # Transform sigma points through observation model
        z_sigmas = np.zeros((n_sigma, m))
        for i in range(n_sigma):
            z_sigmas[i] = self.hx(sigmas_pred[i])

        # Predicted observation mean
        z_pred = np.sum(wm[:, None] * z_sigmas, axis=0)

        # Innovation
        self.innovation = z - z_pred

        # Innovation covariance S = Pzz + R
        S = np.zeros((m, m))
        for i in range(n_sigma):
            dz = z_sigmas[i] - z_pred
            S += wc[i] * np.outer(dz, dz)
        S += self.R
        self.innovation_cov = S

        # Cross-covariance Pxz
        Pxz = np.zeros((n, m))
        for i in range(n_sigma):
            dx = sigmas_pred[i] - self.x
            dz = z_sigmas[i] - z_pred
            Pxz += wc[i] * np.outer(dx, dz)

        # Kalman gain
        try:
            K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(S)

        # State update
        self.x = self.x + K @ self.innovation
        self.P = self.P - K @ S @ K.T

        # Ensure P stays symmetric
        self.P = 0.5 * (self.P + self.P.T)

        # Log-likelihood (for IMM model probability update)
        try:
            sign, logdet = np.linalg.slogdet(S)
            if sign > 0:
                self.log_likelihood = (
                    -0.5 * (m * np.log(2 * np.pi) + logdet
                            + self.innovation @ np.linalg.solve(S, self.innovation))
                )
            else:
                self.log_likelihood = -1e10
        except np.linalg.LinAlgError:
            self.log_likelihood = -1e10
