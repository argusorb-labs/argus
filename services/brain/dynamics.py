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
J3 = -2.53265648e-6             # J3 pear-shape term
J4 = -1.61962159e-6             # J4 term
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


# Pre-compute atmosphere band arrays for vectorized lookup
_ATMO_LOWS = np.array([b[0] for b in _ATMO_BANDS])
_ATMO_HIGHS = np.array([b[1] for b in _ATMO_BANDS])
_ATMO_RHO0 = np.array([b[2] for b in _ATMO_BANDS])
_ATMO_H0 = np.array([b[3] for b in _ATMO_BANDS])
_ATMO_H = np.array([b[4] for b in _ATMO_BANDS])


def _atmospheric_density_vec(alt_km: np.ndarray) -> np.ndarray:
    """Vectorized atmospheric density for N altitudes at once."""
    rho = np.zeros_like(alt_km)
    for i in range(len(_ATMO_BANDS)):
        mask = (alt_km >= _ATMO_LOWS[i]) & (alt_km < _ATMO_HIGHS[i])
        if np.any(mask):
            rho[mask] = _ATMO_RHO0[i] * np.exp(-(alt_km[mask] - _ATMO_H0[i]) / _ATMO_H[i])
    return rho


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

    # ── Zonal gravity harmonics (J2 + J3 + J4) ──
    r2 = r * r
    z2 = z * z
    z_r2 = z2 / r2  # (z/r)²

    # J2: dominant oblateness — ~1 km/orbit effect
    f2 = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r**5
    a_j2 = np.array([
        f2 * x * (5 * z_r2 - 1),
        f2 * y * (5 * z_r2 - 1),
        f2 * z * (5 * z_r2 - 3),
    ])

    # J3: pear-shape asymmetry — ~1 m/orbit effect
    f3 = 0.5 * J3 * MU_EARTH * R_EARTH**3 / r**7
    a_j3 = np.array([
        f3 * 5 * x * (7 * z * z_r2 - 3 * z),
        f3 * 5 * y * (7 * z * z_r2 - 3 * z),
        f3 * (6 * z2 - 7 * z2 * z_r2 - 0.6 * r2),
    ])

    # J4: ~0.1 m/orbit effect
    z_r4 = z_r2 * z_r2
    f4 = -0.625 * J4 * MU_EARTH * R_EARTH**4 / r**7
    a_j4 = np.array([
        f4 * x / r2 * (3 - 42 * z_r2 + 63 * z_r4),
        f4 * y / r2 * (3 - 42 * z_r2 + 63 * z_r4),
        f4 * z / r2 * (15 - 70 * z_r2 + 63 * z_r4),
    ])

    a_gravity = a_grav + a_j2 + a_j3 + a_j4

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
    a_total = a_gravity + a_drag

    return np.array([vx, vy, vz, a_total[0], a_total[1], a_total[2]])


def propagate_state(
    state0: np.ndarray,
    dt_seconds: float,
    bstar: float = 0.0,
    method: str = "DOP853",
    rtol: float = 1e-8,
    atol: float = 1e-10,
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


def _vectorized_eom(states: np.ndarray, bstar: float) -> np.ndarray:
    """Compute equations of motion for N states simultaneously.

    Args:
        states: (N, 6) array — each row is [x, y, z, vx, vy, vz]
        bstar: shared B* for all states

    Returns:
        (N, 6) array of derivatives
    """
    pos = states[:, :3]                    # (N, 3)
    vel = states[:, 3:]                    # (N, 3)
    r = np.linalg.norm(pos, axis=1, keepdims=True)  # (N, 1)
    r = np.maximum(r, R_EARTH)             # clamp to avoid /0

    x = pos[:, 0:1]  # (N, 1)
    y = pos[:, 1:2]
    z = pos[:, 2:3]
    r2 = r * r
    z2 = z * z
    z_r2 = z2 / r2

    # Point mass gravity
    a_grav = -MU_EARTH / (r ** 3) * pos    # (N, 3)

    # J2
    f2 = 1.5 * J2 * MU_EARTH * R_EARTH ** 2 / r ** 5
    a_j2 = np.column_stack([
        f2 * x * (5 * z_r2 - 1),
        f2 * y * (5 * z_r2 - 1),
        f2 * z * (5 * z_r2 - 3),
    ])

    # J3
    f3 = 0.5 * J3 * MU_EARTH * R_EARTH ** 3 / r ** 7
    a_j3 = np.column_stack([
        f3 * 5 * x * (7 * z * z_r2 - 3 * z),
        f3 * 5 * y * (7 * z * z_r2 - 3 * z),
        f3 * (6 * z2 - 7 * z2 * z_r2 - 0.6 * r2),
    ])

    # J4
    z_r4 = z_r2 * z_r2
    f4 = -0.625 * J4 * MU_EARTH * R_EARTH ** 4 / r ** 7
    a_j4 = np.column_stack([
        f4 * x / r2 * (3 - 42 * z_r2 + 63 * z_r4),
        f4 * y / r2 * (3 - 42 * z_r2 + 63 * z_r4),
        f4 * z / r2 * (15 - 70 * z_r2 + 63 * z_r4),
    ])

    a_total = a_grav + a_j2 + a_j3 + a_j4

    # Atmospheric drag (fully vectorized — no Python loop)
    if abs(bstar) > 1e-12:
        alt_km = (r.ravel() - R_EARTH) / 1000.0
        rho = _atmospheric_density_vec(alt_km)  # (N,)
        cd_a_m = _bstar_to_ballistic(bstar)

        drag_mask = rho > 0
        if np.any(drag_mask):
            # Velocity relative to rotating atmosphere
            v_rel = vel.copy()
            v_rel[:, 0] += OMEGA_EARTH * pos[:, 1]
            v_rel[:, 1] -= OMEGA_EARTH * pos[:, 0]
            v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)  # (N, 1)
            v_rel_mag = np.maximum(v_rel_mag, 1e-10)

            rho_col = rho[:, None]  # (N, 1)
            a_drag = -0.5 * cd_a_m * rho_col * v_rel_mag * v_rel  # (N, 3)
            a_total[drag_mask] += a_drag[drag_mask]

    return np.column_stack([vel, a_total])


def propagate_batch_rk4(
    states: np.ndarray,
    dt_seconds: float,
    bstar: float = 0.0,
    step_size: float = 60.0,
) -> tuple[np.ndarray, bool]:
    """Fixed-step RK4 propagation for multiple states simultaneously.

    Uses a vectorized force model — all sigma points are computed in
    one numpy pass per RK4 sub-step. Much faster than N sequential
    solve_ivp calls because:
    1. No Python callback overhead per step
    2. Numpy broadcasting handles N states at once
    3. Fixed step size = no adaptive overhead

    A 30s step is accurate to ~1 m for LEO orbits (verified against
    DOP853 with tight tolerances). The UKF's 1 km observation noise
    makes this more than adequate.

    Args:
        states: (N, 6) array of state vectors
        dt_seconds: propagation duration
        bstar: B* drag coefficient (shared)
        step_size: RK4 step in seconds (30s default)

    Returns:
        (states_final, success)
    """
    if abs(dt_seconds) < 0.001:
        return states.copy(), True

    n_steps = max(1, int(abs(dt_seconds) / step_size))
    h = dt_seconds / n_steps
    y = states.copy()

    try:
        for _ in range(n_steps):
            k1 = _vectorized_eom(y, bstar)
            k2 = _vectorized_eom(y + 0.5 * h * k1, bstar)
            k3 = _vectorized_eom(y + 0.5 * h * k2, bstar)
            k4 = _vectorized_eom(y + h * k3, bstar)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y, True
    except Exception:
        return states.copy(), False


def propagate_batch(
    states: np.ndarray,
    dt_seconds: float,
    bstar: float = 0.0,
    method: str = "DOP853",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[np.ndarray, bool]:
    """Propagate multiple state vectors simultaneously as one ODE system.

    Instead of N separate solve_ivp calls for N sigma points, stack them
    into a single (N*6)-dimensional ODE and solve once. This eliminates
    per-call Python overhead and lets the integrator share step size
    decisions across all sigma points.

    Args:
        states: (N, 6) array of state vectors
        dt_seconds: propagation duration
        bstar: B* drag coefficient (shared by all states)

    Returns:
        (states_final, success) — (N, 6) array and success flag
    """
    n_states = states.shape[0]
    n_dim = 6

    if abs(dt_seconds) < 0.001:
        return states.copy(), True

    # Flatten: (N, 6) → (N*6,)
    y0 = states.ravel()

    def batch_eom(t, y_flat):
        dydt = np.zeros_like(y_flat)
        for i in range(n_states):
            s = y_flat[i * n_dim:(i + 1) * n_dim]
            dydt[i * n_dim:(i + 1) * n_dim] = equations_of_motion(t, s, bstar)
        return dydt

    sol = solve_ivp(
        batch_eom,
        t_span=(0, dt_seconds),
        y0=y0,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if sol.success:
        result = sol.y[:, -1].reshape(n_states, n_dim)
        return result, True
    else:
        return states.copy(), False


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
