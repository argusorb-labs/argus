"""Independent numerical checks: DOP853 vector force law, Cartesian disk quadrature.

Deliberately does not import services.demo_numerics: catches force-law, frame,
quadrature and dense-output implementation errors, not common model assumptions.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize_scalar
from sgp4.api import Satrec, WGS72, jday
from datetime import datetime


def trajectory(state, horizon):
    def rhs(t, y):
        x, yy, z = y[:3]
        r2 = x * x + yy * yy + z * z
        r = np.sqrt(r2)
        central = -398600800000000.0 / r**3
        j = 1.5 * 0.001082616 * 398600800000000.0 * 6378135.0**2 / r**5
        return [
            *y[3:],
            x * (central + j * (5 * z * z / r2 - 1)),
            yy * (central + j * (5 * z * z / r2 - 1)),
            z * (central + j * (5 * z * z / r2 - 3)),
        ]

    solution = solve_ivp(
        rhs,
        (0, horizon),
        state,
        method="DOP853",
        rtol=2e-13,
        atol=1e-7,
        max_step=20,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.sol


def impulse(state, dv):
    r = state[:3] / np.linalg.norm(state[:3])
    c = np.cross(state[:3], state[3:])
    c /= np.linalg.norm(c)
    i = np.cross(c, r)
    return np.r_[state[:3], state[3:] + dv[0] * r + dv[1] * i + dv[2] * c]


def maneuver(state, horizon, burn, dv):
    before = trajectory(state, horizon)
    after = trajectory(impulse(before(burn), dv), horizon - burn)
    return lambda t: before(t) if t < burn else after(t - burn)


def minimum(a, b, horizon, grid=10):
    def distance2(t):
        d = a(t)[:3] - b(t)[:3]
        return float(d @ d)

    times = np.linspace(0, horizon, int(np.ceil(horizon / grid)) + 1)
    values = [distance2(t) for t in times]
    candidates = [(times[0], values[0]), (times[-1], values[-1])]
    for k in range(1, len(times) - 1):
        if values[k] <= min(values[k - 1], values[k + 1]):
            # Optimize seconds relative to bracket: avoids xatol relative to large clock.
            center = times[k]
            fit = minimize_scalar(
                lambda dt: distance2(center + dt),
                bounds=(times[k - 1] - center, times[k + 1] - center),
                method="bounded",
                options={"xatol": 1e-10},
            )
            candidates.append((center + fit.x, fit.fun))
    t, squared = min(candidates, key=lambda x: x[1])
    return {
        "time_s": float(t),
        "distance_m": float(np.sqrt(squared)),
        "relative_speed_mps": float(np.linalg.norm(a(t)[3:] - b(t)[3:])),
    }


def tle_path(record, epoch):
    dt = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
    jd, f = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
    )
    sat = Satrec.twoline2rv(*record["tle"], WGS72)

    def at(t):
        frac = f + t / 86400
        error, r, v = sat.sgp4(jd + np.floor(frac), frac % 1)
        if error:
            raise ValueError(error)
        return np.array([*r, *v]) * 1000

    return at


def probability(a, b, ca, cb, hbr):
    def rotation(y):
        r = y[:3] / np.linalg.norm(y[:3])
        c = np.cross(y[:3], y[3:])
        c /= np.linalg.norm(c)
        return np.array([r, np.cross(c, r), c]).T

    velocity = b[3:] - a[3:]
    normal = velocity / np.linalg.norm(velocity)
    u = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
    u /= np.linalg.norm(u)
    plane = np.array([u, np.cross(normal, u)])
    ra, rb = rotation(a), rotation(b)
    covariance = plane @ (ra @ ca @ ra.T + rb @ cb @ rb.T) @ plane.T
    mean = plane @ (b[:3] - a[:3])
    inverse = np.linalg.inv(covariance)
    scale = 2 * np.pi * np.sqrt(np.linalg.det(covariance))

    def slice_at(x):
        bound = np.sqrt(max(0, hbr * hbr - x * x))

        def density(y):
            d = np.array([x, y]) - mean
            return np.exp(-0.5 * d @ inverse @ d) / scale

        return quad(density, -bound, bound, epsabs=1e-15, epsrel=1e-10)[0]

    pc, error = quad(slice_at, -hbr, hbr, epsabs=1e-14, epsrel=1e-9)
    return {"pc": float(pc), "quadrature_error_estimate": float(error)}
