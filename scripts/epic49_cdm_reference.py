"""Independent Cartesian adaptive integration and TCA geometry reconstruction.

Consumes validated normalized source inputs, imports no production math.
Does not independently verify source interpretation or common model assumptions.
"""

import numpy as np
from scipy.integrate import quad


def evaluate(cdm):
    states, covariances, sensitivities = [], [], []
    for obj in cdm["objects"]:
        y = np.array(obj["state_itrf_si"])
        x, yy, z, vx, vy, vz = y
        # Different implementation of derivative transformation; nominal spin.
        inertial = np.array(
            [vx - 7.29211514670698e-5 * yy, vy + 7.29211514670698e-5 * x, vz]
        )
        radial = np.array([x, yy, z]) / np.linalg.norm(y[:3])
        transverse = inertial - np.dot(inertial, radial) * radial
        transverse /= np.linalg.norm(transverse)
        orbit_normal = np.cross(radial, transverse)
        rotation = np.array([radial, transverse, orbit_normal]).T
        c = np.array(obj["covariance_rtn_si"])[:3, :3]
        covariances.append(rotation.dot(c).dot(rotation.T))
        states.append(np.r_[y[:3], inertial])
        g = obj["sensitivity_position_rtn_m"]
        sensitivities.append(None if g is None else rotation.dot(g))
    a, b = states
    d = b[:3] - a[:3]
    w = b[3:] - a[3:]
    w /= np.linalg.norm(w)
    # Put the miss along first encounter axis (different from production plane).
    first = d - w * np.dot(d, w)
    first /= np.linalg.norm(first)
    projection = np.array([first, np.cross(w, first)])
    mean = projection.dot(d)
    total = covariances[0] + covariances[1]

    def integrate(c):
        projected = projection.dot(c).dot(projection.T)
        inverse = np.linalg.inv(projected)
        normalization = 2 * np.pi * np.sqrt(np.linalg.det(projected))
        radius = cdm["hbr_m"]
        inner_errors = []

        def slice_at(x):
            limit = np.sqrt(max(0, radius * radius - x * x))

            def density(y):
                delta = np.array([x, y]) - mean
                return np.exp(-0.5 * delta.dot(inverse).dot(delta)) / normalization

            value, error = quad(density, -limit, limit, epsabs=1e-16, epsrel=1e-11)
            inner_errors.append(error)
            return value

        value, error = quad(slice_at, -radius, radius, epsabs=1e-14, epsrel=1e-11)
        return {
            "pc": float(value),
            "quadrature_error_estimate": float(error),
            "inner_error_bound_estimate": float(2 * radius * max(inner_errors)),
            "eigenvalues_m2": np.linalg.eigvalsh(c).tolist(),
            "determinant_m6": float(np.linalg.det(c)),
            "mean_plane_m": mean.tolist(),
            "covariance_plane_m2": projected.tolist(),
        }

    result = {"baseline": integrate(total)}
    sigmas = [obj["density_sigma"] for obj in cdm["objects"]]
    if all(s is not None for s in sigmas + sensitivities):
        p, s = sensitivities
        # Sum dyads in reference rather than using production cross matrix.
        adjusted = total.copy()
        adjusted -= sigmas[0] * sigmas[1] * np.einsum("i,j->ij", p, s)
        adjusted -= sigmas[0] * sigmas[1] * np.einsum("i,j->ij", s, p)
        result["correlation_sensitivity"] = integrate(adjusted)
    return result
