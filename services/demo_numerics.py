"""Bounded research numerics; SI units, frozen TEME axes, WGS72 two-body/J2.

No drag, Earth rotation, covariance transport, commands or flight-authority claims.
RIC position covariances are supplied *at each TCA*, independently for each body.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp
from sgp4.api import Satrec, WGS72, jday

MU = 398600.8e9
EARTH_RADIUS_M = 6378135.0
J2 = 0.001082616


def km_state_to_si(position_km, velocity_kmps):
    return np.r_[position_km, velocity_kmps].astype(float) * 1000.0


def checked_state(state):
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("state must be six finite SI components")
    return state


def ric_basis(state):
    state = checked_state(state)
    r, v = state[:3], state[3:]
    h = np.cross(r, v)
    if np.linalg.norm(r) == 0 or np.linalg.norm(h) == 0:
        raise ValueError("undefined RIC axes")
    radial = r / np.linalg.norm(r)
    cross = h / np.linalg.norm(h)
    return np.column_stack((radial, np.cross(cross, radial), cross))


def apply_impulse(state, dv_ric_mps):
    result = checked_state(state).copy()
    dv = np.asarray(dv_ric_mps, dtype=float)
    if dv.shape != (3,) or not np.isfinite(dv).all():
        raise ValueError("impulse must be three finite SI components")
    result[3:] += ric_basis(result) @ dv
    return result


def derivative(state):
    r = state[:3]
    norm = np.linalg.norm(r)
    if norm <= EARTH_RADIUS_M:
        raise ValueError("trajectory intersects Earth; model inapplicable")
    q = (r[2] / norm) ** 2
    perturbation = 1.5 * J2 * MU * EARTH_RADIUS_M**2 / norm**5
    acceleration = -MU * r / norm**3 + perturbation * r * np.array(
        [5 * q - 1, 5 * q - 1, 5 * q - 3]
    )
    return np.r_[state[3:], acceleration]


def trajectory(state, duration_s, step_s=5.0):
    """Classical fixed-step RK4 with cubic Hermite dense states (supports backward)."""
    state = checked_state(state).copy()
    if (
        not np.isfinite(duration_s)
        or duration_s == 0
        or not np.isfinite(step_s)
        or step_s <= 0
    ):
        raise ValueError("finite nonzero duration and positive step required")
    count = int(np.ceil(abs(duration_s) / step_s))
    times = np.linspace(0, duration_s, count + 1)
    states = [state.copy()]
    for dt in np.diff(times):
        k1 = derivative(state)
        k2 = derivative(state + dt * k1 / 2)
        k3 = derivative(state + dt * k2 / 2)
        k4 = derivative(state + dt * k3)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(state.copy())
    states = np.array(states)
    slopes = np.array([derivative(s) for s in states])
    if duration_s < 0:
        times, states, slopes = times[::-1], states[::-1], slopes[::-1]
    spline = CubicHermiteSpline(times, states, slopes, extrapolate=False)
    return lambda t: np.asarray(spline(t))


def maneuver_trajectory(state, horizon_s, burn_time_s, dv_ric_mps, step_s=5):
    if not 0 <= burn_time_s < horizon_s:
        raise ValueError("burn outside horizon")
    pre = trajectory(state, horizon_s, step_s)
    post = trajectory(
        apply_impulse(pre(burn_time_s), dv_ric_mps), horizon_s - burn_time_s, step_s
    )
    return lambda t: pre(t) if t < burn_time_s else post(t - burn_time_s)


def closest_approach(primary, secondary, start_s, end_s, grid_s=30):
    """Refine every derivative sign bracket, plus distance brackets and endpoints.

    Grid is a bracket finder, never a distance gate. This bounded orbit screen
    assumes isolated minima resolvable at grid_s; convergence is checked separately.
    """
    if not 0 < grid_s or not start_s < end_s:
        raise ValueError("invalid search interval")

    def squared(t):
        d = secondary(t)[:3] - primary(t)[:3]
        return float(d @ d)

    def rate(t):
        d = secondary(t) - primary(t)
        return float(d[:3] @ d[3:])

    grid = np.linspace(start_s, end_s, int(np.ceil((end_s - start_s) / grid_s)) + 1)
    values = [squared(t) for t in grid]
    rates = [rate(t) for t in grid]
    candidates = [start_s, end_s]
    for i in range(len(grid) - 1):
        if rates[i] <= 0 <= rates[i + 1]:
            candidates.append(brentq(rate, grid[i], grid[i + 1], xtol=1e-8))
    for i in range(1, len(grid) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            fit = minimize_scalar(
                squared,
                bounds=(grid[i - 1], grid[i + 1]),
                method="bounded",
                options={"xatol": 1e-8},
            )
            candidates.append(float(fit.x))
    t = min(candidates, key=squared)
    a, b = primary(t), secondary(t)
    return {
        "time_s": float(t),
        "distance_m": float(np.linalg.norm(b[:3] - a[:3])),
        "relative_speed_mps": float(np.linalg.norm(b[3:] - a[3:])),
        "boundary_minimum": bool(t == start_s or t == end_s),
    }


def tle_trajectory(record, epoch_utc):
    satellite = Satrec.twoline2rv(*record["tle"], WGS72)
    dt = datetime.fromisoformat(epoch_utc.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    jd, fraction = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
    )

    def at(t):
        # Keep sub-second offsets out of the large Julian-date float.
        f = fraction + float(t) / 86400
        whole = np.floor(f)
        error, r, v = satellite.sgp4(jd + whole, f - whole)
        if error:
            raise ValueError(f"SGP4 propagation error {error}: {record['id']}")
        return km_state_to_si(r, v)

    return at


def historical_geometry(case, alternate=False):
    b = case["alternate_secondary"] if alternate else case["secondary"]
    return closest_approach(
        tle_trajectory(case["primary"], case["search_epoch_utc"]),
        tle_trajectory(b, case["search_epoch_utc"]),
        0,
        120,
    )


def valid_covariance(covariance):
    c = np.asarray(covariance, dtype=float)
    if (
        c.shape != (3, 3)
        or not np.isfinite(c).all()
        or not np.allclose(c, c.T, rtol=0, atol=1e-10)
    ):
        raise ValueError("covariance must be finite symmetric 3x3")
    np.linalg.cholesky(c)
    return c


def encounter_projection(a, b, cov_a, cov_b):
    """Each object's own RIC → common axes → relative-velocity encounter plane."""
    velocity = b[3:] - a[3:]
    normal = velocity / np.linalg.norm(velocity)
    helper = np.eye(3)[np.argmin(abs(normal))]
    x = np.cross(normal, helper)
    x /= np.linalg.norm(x)
    plane = np.column_stack((x, np.cross(normal, x)))
    ra, rb = ric_basis(a), ric_basis(b)
    common = ra @ cov_a @ ra.T + rb @ cov_b @ rb.T
    return plane.T @ (b[:3] - a[:3]), plane.T @ common @ plane


def disk_probability(mean, covariance, hbr_m, order=64):
    """Polar Gaussian disk integral, log-sum-exp including tiny probabilities."""
    if not np.isfinite(hbr_m) or hbr_m <= 0:
        raise ValueError("positive finite summed hard-body radius required")
    nodes, weights = leggauss(order)
    radial = (nodes + 1) * hbr_m / 2
    angles = (nodes + 1) * np.pi
    x = np.stack(
        np.broadcast_arrays(
            radial[:, None] * np.cos(angles), radial[:, None] * np.sin(angles)
        ),
        axis=-1,
    )
    delta = x - np.asarray(mean)
    exponent = -0.5 * np.einsum(
        "...i,ij,...j->...", delta, np.linalg.inv(covariance), delta
    )
    log_weights = np.log(
        weights[:, None] * weights[None, :] * radial[:, None] * hbr_m / 2 * np.pi
    )
    log_pc = float(
        logsumexp(exponent + log_weights)
        - np.log(2 * np.pi)
        - 0.5 * np.linalg.slogdet(covariance)[1]
    )
    return {
        "pc": float(np.exp(log_pc)),
        "log_pc": log_pc,
        "underflow": bool(log_pc < np.log(np.nextafter(0.0, 1.0))),
    }


def encounter_probability(
    a, b, cov_a, cov_b, hbr_m, assumed=True, screen_m=50000, boundary=False
):
    a, b = checked_state(a), checked_state(b)
    result = {
        "pc": None,
        "log_pc": None,
        "covariance_mode": "assumed_at_tca" if assumed else "missing",
        "hbr_m": float(hbr_m),
    }
    if np.linalg.norm(b[3:] - a[3:]) <= 1000:
        return dict(result, status="low_relative_speed")
    if boundary:
        return dict(result, status="boundary_minimum_inapplicable")
    if np.linalg.norm(b[:3] - a[:3]) > screen_m:
        return dict(result, status="outside_screened_volume")
    if not assumed or cov_a is None or cov_b is None:
        return dict(result, status="missing_covariance")
    try:
        ca, cb = valid_covariance(cov_a), valid_covariance(cov_b)
        mean, covariance = encounter_projection(a, b, ca, cb)
    except (ValueError, np.linalg.LinAlgError):
        return dict(result, status="invalid_covariance")
    speed = np.linalg.norm(b[3:] - a[3:])
    duration = (hbr_m + 3 * np.sqrt(np.linalg.eigvalsh(covariance).max())) / speed
    # Brief linear encounter compared with local dynamical time; longitudinal miss must be small.
    dynamical_time = np.sqrt(
        min(np.linalg.norm(a[:3]), np.linalg.norm(b[:3])) ** 3 / MU
    )
    longitudinal = abs(np.dot(b[:3] - a[:3], (b[3:] - a[3:]) / speed))
    if duration > 0.01 * dynamical_time or longitudinal > 1:
        return dict(result, status="nonlinear_encounter_inapplicable")
    return dict(
        result,
        **disk_probability(mean, covariance, hbr_m),
        status="evaluated_assumed",
        linear_encounter_duration_s=float(duration),
    )


def evaluate_case(case, step_s=5, grid_s=30):
    """Evaluate supplied vectors; IDs are display labels only. Finite scope only."""
    horizon = case["horizon_s"]
    primary, secondary = case["primary"], case["secondary"]
    bodies = [secondary] + case["catalog"]
    for body in [primary] + bodies:
        if body["epoch_utc"] != case["epoch_utc"] or body["frame"] != case["frame"]:
            raise ValueError("mixed epoch/frame")
        if not np.isfinite(body["radius_m"]) or body["radius_m"] <= 0:
            raise ValueError("invalid body radius")
    paths = {
        body["id"]: trajectory(body["state_si"], horizon, step_s) for body in bodies
    }
    if len(paths) != len(bodies):
        raise ValueError("duplicate body IDs")
    results = []
    thresholds = case["thresholds"]
    for option in case["options"]:
        path = maneuver_trajectory(
            primary["state_si"],
            horizon,
            option["burn_time_s"],
            option["dv_ric_mps"],
            step_s,
        )
        encounters = []
        for body in bodies:
            other = paths[body["id"]]
            encounter = closest_approach(path, other, 0, horizon, grid_s)
            t = encounter["time_s"]
            probability = encounter_probability(
                path(t),
                other(t),
                primary.get("covariance_ric_m2"),
                body.get("covariance_ric_m2"),
                primary["radius_m"] + body["radius_m"],
                assumed=case["assumed_scenario"],
                screen_m=thresholds["screen_m"],
                boundary=encounter["boundary_minimum"],
            )
            pc = probability["pc"]
            encounter.update(
                object_id=body["id"],
                probability=probability,
                alert=pc is not None and pc > thresholds["alert_pc"],
                rejected=pc is not None and pc > thresholds["reject_pc"],
            )
            encounters.append(encounter)
        magnitude = float(np.linalg.norm(option["dv_ric_mps"]))
        rejected = magnitude > thresholds["max_dv_mps"] or any(
            e["rejected"] for e in encounters
        )
        if rejected:
            evaluation_status = "rejected_assumed_constraints"
        elif encounters[0]["probability"]["pc"] is None:
            evaluation_status = "abstain_missing_primary_probability"
        elif any(
            e["distance_m"] <= thresholds["screen_m"] and e["probability"]["pc"] is None
            for e in encounters[1:]
        ):
            evaluation_status = "abstain_missing_secondary_probability"
        else:
            evaluation_status = "eligible_for_draft_within_assumed_scope"
        results.append(
            {
                "id": option["id"],
                "delta_v_mps": magnitude,
                "primary_encounter": encounters[0],
                "secondary_encounters": encounters[1:],
                "rejected": rejected,
                "evaluation_status": evaluation_status,
                "recommendation_status": "DRAFT_PENDING_HUMAN_APPROVAL",
            }
        )
    return {
        "options": results,
        "screen_scope": {
            "catalog_objects": len(case["catalog"]),
            "horizon_s": horizon,
            "radius_m": thresholds["screen_m"],
            "global_safety_claim": False,
            "conjunctions_inside_volume": len(
                {
                    e["object_id"]
                    for option in results
                    for e in option["secondary_encounters"]
                    if e["distance_m"] <= thresholds["screen_m"]
                }
            ),
            "meaning": "bounded trajectory minima; missing Pc remains not evaluated",
        },
    }
