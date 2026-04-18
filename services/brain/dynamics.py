"""Orbital dynamics — force model and numerical propagator.

Provides the state transition function for the UKF: given a state
[x, y, z, vx, vy, vz] in the TEME frame, propagate it forward in
time under the influence of gravity (point mass + J2) and atmospheric
drag (exponential atmosphere parameterized by B*).

This is deliberately a simplified force model — accurate enough for
anomaly detection and covariance propagation, but not for operational
orbit determination. The force model can be extended later with J3/J4,
lunisolar perturbations, solar radiation pressure, etc.

Coordinate frame: TEME (True Equator Mean Equinox) — the native frame
of SGP4/TLE. Working in TEME avoids coordinate transform overhead
since our observations (TLE → sgp4) are already in TEME.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# ── Constants ──

MU_EARTH = 3.986004418e14       # m³/s² — gravitational parameter
R_EARTH = 6.3781363e6           # m — equatorial radius
J2 = 1.08262668e-3              # J2 oblateness coefficient
OMEGA_EARTH = 7.2921159e-5      # rad/s — Earth rotation rate

# Exponential atmosphere model (simplified CIRA/MSIS approximation)
# Piecewise: below/above 500 km use different scale heights
# Reference: Vallado, "Fundamentals of Astrodynamics", Table 8-4
# Calibrated for moderate solar activity (F10.7 ~ 150).
# Values from NRLMSISE-00 typical outputs — NOT a minimum/maximum model,
# but a "reasonable middle" that the UKF's process noise can absorb.
_ATMO_BANDS = [
    # (alt_km_low, alt_km_high, rho0_kg_m3, h0_km, H_km)
    (0,    100,   1.225,        0,      8.5),
    (100,  150,   5.0e-7,     100,      6.0),
    (150,  200,   2.5e-10,    150,     22.0),
    (200,  250,   2.5e-10,    200,     30.0),
    (250,  300,   4.0e-11,    250,     35.0),
    (300,  400,   2.0e-11,    300,     45.0),
    (400,  500,   4.0e-12,    400,     55.0),
    (500,  600,   5.0e-13,    500,     65.0),
    (600,  700,   8.0e-14,    600,     75.0),
    (700,  1000,  1.5e-14,    700,     90.0),
    (1000, 2000,  3.0e-15,   1000,    150.0),
]


def _atmospheric_density(alt_km: float) -> float:
    """Exponential atmosphere density at given altitude [kg/m³]."""
    if alt_km < 0 or alt_km > 2000:
        return 0.0
    for low, high, rho0, h0, H in _ATMO_BANDS:
        if low <= alt_km < high:
            return rho0 * np.exp(-(alt_km - h0) / H)
    return 0.0


def _bstar_to_ballistic(bstar: float) -> float:
    """Convert TLE B* to effective Cd*A/m for our atmosphere model.

    B* is defined in SGP4's internal units (1/earth_radii) and is
    calibrated against SGP4's own density model. For our exponential
    atmosphere, we need to re-calibrate.

    Empirical: a typical Starlink (Cd~2.2, A~5m², m~260kg →
    Cd*A/m ≈ 0.042 m²/kg) has B* ≈ 1e-4. So:
      Cd*A/m ≈ B* × 420 m²/kg per B*-unit

    This calibration constant absorbs the density reference difference
    between SGP4's model and ours. It will be refined as we accumulate
    UKF residuals; for now it produces the right order of magnitude for
    altitude decay rates (~1-5 km/day for a Starlink without thrust).
    """
    BSTAR_TO_CDAM = 420.0  # empirical calibration
    return bstar * BSTAR_TO_CDAM


def equations_of_motion(t: float, state: np.ndarray, bstar: float = 0.0) -> np.ndarray:
    """Right-hand side of the orbital equations of motion.

    Args:
        t: time (seconds from epoch) — unused for autonomous system,
           included for solve_ivp compatibility
        state: [x, y, z, vx, vy, vz] in TEME [m, m/s]
        bstar: TLE B* drag coefficient

    Returns:
        [vx, vy, vz, ax, ay, az] — state derivative
    """
    x, y, z, vx, vy, vz = state
    r_vec = state[:3]
    v_vec = state[3:]
    r = np.linalg.norm(r_vec)

    if r < R_EARTH:
        # Inside Earth — return zeros to avoid NaN (integration will stop)
        return np.zeros(6)

    # ── Point mass gravity ──
    a_grav = -MU_EARTH / r**3 * r_vec

    # ── J2 oblateness perturbation ──
    r2 = r * r
    z2 = z * z
    factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r**5
    a_j2 = np.array([
        factor * x * (5 * z2 / r2 - 1),
        factor * y * (5 * z2 / r2 - 1),
        factor * z * (5 * z2 / r2 - 3),
    ])

    # ── Atmospheric drag ──
    alt_km = (r - R_EARTH) / 1000.0
    rho = _atmospheric_density(alt_km)
    a_drag = np.zeros(3)
    if rho > 0 and abs(bstar) > 1e-12:
        # Velocity relative to rotating atmosphere
        v_atm = np.array([
            vx + OMEGA_EARTH * y,
            vy - OMEGA_EARTH * x,
            vz,
        ])
        v_rel = v_vec - np.array([-OMEGA_EARTH * y, OMEGA_EARTH * x, 0.0])
        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag > 0:
            cd_a_over_m = _bstar_to_ballistic(bstar)
            a_drag = -0.5 * cd_a_over_m * rho * v_rel_mag * v_rel

    # Total acceleration
    a_total = a_grav + a_j2 + a_drag

    return np.array([vx, vy, vz, a_total[0], a_total[1], a_total[2]])


def propagate_state(
    state0: np.ndarray,
    dt_seconds: float,
    bstar: float = 0.0,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, bool]:
    """Propagate a state vector forward by dt_seconds.

    Args:
        state0: [x, y, z, vx, vy, vz] in TEME [m, m/s]
        dt_seconds: propagation duration (positive = forward)
        bstar: TLE B* for drag computation
        method: ODE solver (DOP853 recommended for orbital mechanics)
        rtol, atol: integration tolerances

    Returns:
        (state_final, success) — final state and whether integration succeeded
    """
    if abs(dt_seconds) < 0.001:
        return state0.copy(), True

    sol = solve_ivp(
        equations_of_motion,
        t_span=(0, dt_seconds),
        y0=state0,
        method=method,
        rtol=rtol,
        atol=atol,
        args=(bstar,),
        dense_output=False,
    )

    if sol.success:
        return sol.y[:, -1], True
    else:
        return state0.copy(), False


def tle_to_state(line1: str, line2: str, epoch_offset_min: float = 0.0) -> np.ndarray | None:
    """Convert TLE lines to a state vector [x, y, z, vx, vy, vz] in TEME.

    Uses sgp4 to propagate to the TLE epoch (+ optional offset in minutes).
    Returns position in meters, velocity in m/s, or None if sgp4 fails.
    """
    from sgp4.api import Satrec, WGS72

    sat = Satrec.twoline2rv(line1, line2, WGS72)
    e, r, v = sat.sgp4(sat.jdsatepoch, sat.jdsatepochF + epoch_offset_min / 1440.0)

    if e != 0:
        return None

    # sgp4 returns km and km/s — convert to m and m/s
    state = np.array([
        r[0] * 1000, r[1] * 1000, r[2] * 1000,
        v[0] * 1000, v[1] * 1000, v[2] * 1000,
    ])
    return state
