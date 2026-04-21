"""Physics-based state predictor for pre-computing innovation features.

Design: the ML model consumes innovation = observed − f_physics as an
input feature, rather than computing f_physics inside its forward pass.
This keeps physics fully frozen (no gradient leakage, v0.5 insight #2)
and makes training faster.

Innovation definition (SGP4-based):

    innovation[t] = SGP4(TLE[t], epoch_t) − SGP4(TLE[t-1], epoch_t)

That is, evaluate both consecutive TLEs at the *newer* TLE's epoch and
take the difference. For a satellite in clean station-keeping, both
TLEs are fits to the same underlying trajectory, so the difference is
only orbit-determination noise (a few hundred meters). An anomaly
(maneuver, breakup, attitude change) makes TLE[t] inconsistent with
TLE[t-1]'s SGP4 propagation, producing a km-scale innovation spike.

Why SGP4 and not a numerical J2+J3+J4+drag integrator: TLEs are
themselves SGP4 fits, so the "true" propagator for anomaly detection
is SGP4 (self-consistent baseline). A third-party integrator adds
5 km of model error per day which drowns the anomaly signal for
typical maneuver magnitudes. Also: SGP4 is C-compiled and runs in
~3 μs vs ~100 ms for a batched RK4 step, 3-4 orders of magnitude
faster for mass preprocessing.
"""

from __future__ import annotations

import numpy as np
from sgp4.api import Satrec

from services.brain.dynamics import propagate_batch_rk4, tle_to_state


def propagate_one_step(
    state: np.ndarray,
    dt_seconds: float,
    bstar: float = 0.0,
) -> np.ndarray:
    """Propagate a single 6D state forward by dt seconds under full
    J2+J3+J4+drag dynamics. Returns the predicted state."""
    states_1 = state.reshape(1, 6)
    pred, ok = propagate_batch_rk4(states_1, dt_seconds, bstar=bstar)
    if not ok:
        # Propagator failed — return the current state as prediction
        # so innovation = 0 and the ML model ignores this sample
        return state.copy()
    return pred[0]


def compute_innovation_series_sgp4(
    line1_list: list[str],
    line2_list: list[str],
) -> np.ndarray:
    """Compute innovation sequence using SGP4 on consecutive TLEs.

    For each t >= 1:
        innovation[t] = SGP4(TLE[t], epoch_t) − SGP4(TLE[t-1], epoch_t)

    Evaluated at TLE[t]'s own epoch. TLEs must be sorted chronologically.

    Normal station-keeping: innovation is orbit-determination noise only.
    Maneuver / breakup / anomaly: TLE[t-1]'s SGP4 prediction at epoch_t
    disagrees with TLE[t]'s fit at the same time, producing a km-scale
    innovation spike.

    Returns:
        innovations: (T, 6) in meters and m/s (TEME frame).
                     innovation[0] is zeros.
    """
    T = len(line1_list)
    assert len(line2_list) == T

    innovations = np.zeros((T, 6))
    if T < 2:
        return innovations

    # Pre-build Satrec objects — ~3 μs each
    sats: list[Satrec | None] = []
    for l1, l2 in zip(line1_list, line2_list):
        try:
            sats.append(Satrec.twoline2rv(l1, l2))
        except Exception:
            sats.append(None)

    for t in range(1, T):
        if sats[t] is None or sats[t - 1] is None:
            continue
        # Use TLE[t]'s own epoch as evaluation time
        jd = sats[t].jdsatepoch
        fr = sats[t].jdsatepochF

        # TLE[t] at its own epoch (the "observed" state)
        e_t, r_t, v_t = sats[t].sgp4(jd, fr)
        # TLE[t-1] propagated forward to TLE[t]'s epoch
        e_prev, r_prev, v_prev = sats[t - 1].sgp4(jd, fr)
        if e_t != 0 or e_prev != 0:
            continue

        # km / km/s → m / m/s
        innovations[t, 0] = (r_t[0] - r_prev[0]) * 1000
        innovations[t, 1] = (r_t[1] - r_prev[1]) * 1000
        innovations[t, 2] = (r_t[2] - r_prev[2]) * 1000
        innovations[t, 3] = (v_t[0] - v_prev[0]) * 1000
        innovations[t, 4] = (v_t[1] - v_prev[1]) * 1000
        innovations[t, 5] = (v_t[2] - v_prev[2]) * 1000

    return innovations


def compute_innovation_series(
    states: np.ndarray,
    dts_seconds: np.ndarray,
    bstars: np.ndarray,
) -> np.ndarray:
    """[Legacy] J2+J3+J4+drag integrator-based innovation, batched with
    median-dt approximation. Retained for reference but NOT the recommended
    path for v0.6 preprocessing — see compute_innovation_series_sgp4().

    Accuracy degrades sharply when consecutive TLE intervals vary (max
    observed >80h for Starlink historical data), so median-dt batching
    produces large spurious innovations. Use the SGP4 version instead.
    """
    T = states.shape[0]
    innovations = np.zeros_like(states)
    if T < 2:
        return innovations
    starts = states[:-1]
    valid_dts = dts_seconds[1:][dts_seconds[1:] > 0]
    if len(valid_dts) == 0:
        return innovations
    dt_shared = float(np.median(valid_dts))
    bstar_shared = float(np.median(bstars[:-1]))
    predicted, ok = propagate_batch_rk4(starts, dt_shared, bstar=bstar_shared)
    if ok:
        innovations[1:] = states[1:] - predicted
    return innovations


def tle_sequence_to_states(line1_seq: list[str], line2_seq: list[str]) -> np.ndarray:
    """Convert a sequence of TLEs to Cartesian state vectors.

    Each TLE is evaluated at its own epoch (no propagation here — just
    the position/velocity implied by that TLE's Keplerian elements).
    Failed conversions return zeros; caller should filter.

    Returns:
        states: (T, 6) in meters / m/s, TEME frame
    """
    T = len(line1_seq)
    states = np.zeros((T, 6))
    for i, (l1, l2) in enumerate(zip(line1_seq, line2_seq)):
        s = tle_to_state(l1, l2)
        if s is not None:
            states[i] = s
    return states
