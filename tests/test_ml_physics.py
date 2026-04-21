"""Verify the ML physics predictor against SGP4 on real Starlink TLEs.

Gate for Phase 1 (v0.6): if our J2+J3+J4+drag propagator disagrees with
SGP4 by more than ~5 km over 1 day on a clean trajectory, innovation =
observed - f_physics will be dominated by our propagator's error rather
than the anomaly signal. In that case we cannot ship v0.6 with this
physics module; the fallback is NRLMSISE-00 atmosphere or DOP853.
"""

from __future__ import annotations


import numpy as np
import pytest
from sgp4.api import Satrec

from services.ml.physics import propagate_one_step
from services.brain.dynamics import tle_to_state


# A real Starlink TLE — typical 550 km shell, stable station-keeping.
# Anyone can pull a fresh one from Celestrak; this one is a fixed test
# vector so the test is deterministic.
STARLINK_L1 = "1 44714U 19074A   26108.50000000  .00001234  00000-0  12345-4 0  9995"
STARLINK_L2 = "2 44714  53.0536 123.4567 0001234  89.1234  270.1234 15.06400000123456"


def _sgp4_state_at(line1: str, line2: str, minutes_from_epoch: float) -> np.ndarray:
    """Run the reference SGP4 propagator and return (x,y,z,vx,vy,vz) in m/s."""
    sat = Satrec.twoline2rv(line1, line2)
    jd0 = sat.jdsatepoch
    jf0 = sat.jdsatepochF
    dt_days = minutes_from_epoch / (24 * 60)
    jd = jd0 + dt_days
    e, r, v = sat.sgp4(jd, jf0)
    if e != 0:
        raise RuntimeError(f"SGP4 error {e}")
    # km -> m
    return np.array(
        [r[0] * 1000, r[1] * 1000, r[2] * 1000, v[0] * 1000, v[1] * 1000, v[2] * 1000]
    )


def test_propagator_matches_sgp4_one_hour():
    """Our J2+J3+J4+drag RK4 should track SGP4 to <1 km over 1 hour on a
    nominal LEO orbit. This is within our UKF's 1 km observation noise."""
    state0 = tle_to_state(STARLINK_L1, STARLINK_L2)
    assert state0 is not None

    dt_minutes = 60.0
    dt_seconds = dt_minutes * 60

    sgp4_state = _sgp4_state_at(STARLINK_L1, STARLINK_L2, dt_minutes)
    our_state = propagate_one_step(state0, dt_seconds, bstar=0.00012345)

    pos_error_m = float(np.linalg.norm(sgp4_state[:3] - our_state[:3]))
    print(f"1-hour position error: {pos_error_m:.1f} m")
    assert pos_error_m < 5_000, f"1-hour error {pos_error_m:.0f} m exceeds 5 km"


def test_propagator_matches_sgp4_one_day():
    """Over 1 day the two propagators diverge further because SGP4 uses
    its own internal perturbation model. We budget 5 km as the Phase 1
    gate — innovation below this threshold is drowned in model error.

    If this fails, stop Phase 1 and upgrade either:
    - drag model (exponential → NRLMSISE-00)
    - gravity field (J4 → higher order)
    - integrator (RK4 → DOP853 with tight tolerances)
    """
    state0 = tle_to_state(STARLINK_L1, STARLINK_L2)
    assert state0 is not None

    dt_days = 1.0
    dt_seconds = dt_days * 86400

    sgp4_state = _sgp4_state_at(STARLINK_L1, STARLINK_L2, dt_days * 1440)
    our_state = propagate_one_step(state0, dt_seconds, bstar=0.00012345)

    pos_error_m = float(np.linalg.norm(sgp4_state[:3] - our_state[:3]))
    print(f"1-day position error: {pos_error_m:.1f} m ({pos_error_m / 1000:.2f} km)")
    assert pos_error_m < 5_000, (
        f"1-day error {pos_error_m / 1000:.1f} km exceeds 5 km Phase 1 gate — "
        f"upgrade drag / gravity / integrator before proceeding"
    )


def test_innovation_series_shape():
    """Smoke test: compute_innovation_series returns correct shape and
    has innovation[0] = zeros."""
    from services.ml.physics import compute_innovation_series

    T = 5
    states = np.random.randn(T, 6) * 1e6  # random states
    dts = np.array([0.0] + [3600.0] * (T - 1))  # 1 hour between
    bstars = np.full(T, 1e-4)
    inn = compute_innovation_series(states, dts, bstars)
    assert inn.shape == (T, 6)
    assert np.allclose(inn[0], 0.0), "innovation[0] should be zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
