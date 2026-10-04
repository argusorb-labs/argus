"""Bounded NASA CDM risk reconstruction at TCA; no orbit propagation.

ITRF coordinate derivatives become physical velocities via v + omega cross r.
All vectors use a common instantaneous inertial orientation aligned with ITRF
at TCA (not a time series). See fixture README for derivation and limitations.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np

from services.demo_numerics import disk_probability

EARTH_OMEGA_RAD_S = 7.29211514670698e-5
AXES = ("R", "T", "N", "RDOT", "TDOT", "NDOT")
STATE_KEYS = ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
DCP_SIGMA = "DCP Density Forecast Uncertainty"
DCP_POS = "DCP Sensitivity Vector RTN Pos"
DCP_VEL = "DCP Sensitivity Vector RTN Vel"


def _numbers(value, unit, count=1):
    match = re.fullmatch(r"(.*?)\s*\[([^\]]+)\]", value)
    if unit is None:
        text = value
        if "[" in text:
            raise ValueError("unexpected units")
    elif match and match[2] == unit:
        text = match[1]
    else:
        raise ValueError(f"expected explicit unit {unit}: {value}")
    try:
        numbers = np.array([float(v) for v in text.split()])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid numerical field") from exc
    if numbers.shape != (count,) or not np.isfinite(numbers).all():
        raise ValueError("missing or nonfinite numerical field")
    return float(numbers[0]) if count == 1 else numbers.tolist()


def _covariance(value, size):
    c = np.asarray(value, dtype=float)
    if c.shape != (size, size) or not np.isfinite(c).all():
        raise ValueError("missing or nonfinite covariance")
    if not np.allclose(c, c.T, atol=1e-12, rtol=0):
        raise ValueError("asymmetric covariance")
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    return c


def _symmetric(c):
    """Remove only multiplication roundoff from analytically symmetric results."""
    return (c + c.T) / 2


def _positive(value, name):
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return float(value)


def _vector(value, size):
    v = np.asarray(value, dtype=float)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError("missing or nonfinite vector")
    return v


def parse_cdm(text):
    """Parse this released KVN subset, preserving original decimal tokens/comments.

    Fold only value continuation lines. Unknown units/frames and incomplete
    covariance fail closed. This is not a general CCSDS message validator.
    """
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            records.append(line)
        elif records:
            records[-1] += " " + line
        else:
            raise ValueError("orphan continuation")
    sections = [{"fields": {}, "comments": []}]
    for line in records:
        key, value = (x.strip() for x in line.split("=", 1))
        comment = key.startswith("COMMENT ")
        if key == "OBJECT":
            if value != f"OBJECT{len(sections)}" or len(sections) > 2:
                raise ValueError("expected exactly OBJECT1 then OBJECT2")
            sections.append({"fields": {}, "comments": []})
        if comment:
            sections[-1]["comments"].append(line)
            key = key.removeprefix("COMMENT ")
        fields = sections[-1]["fields"]
        if key in fields:
            raise ValueError(f"duplicate field: {key}")
        fields[key] = value
    if len(sections) != 3:
        raise ValueError("exactly two objects required")
    head = sections[0]["fields"]
    try:
        if head["CCSDS_CDM_VERS"] != "1.0":
            raise ValueError("unsupported CDM version")
        for key in ("CREATION_DATE", "TCA"):
            datetime.fromisoformat(head[key])
        if datetime.fromisoformat(head["CREATION_DATE"]) >= datetime.fromisoformat(
            head["TCA"]
        ):
            raise ValueError("creation must precede TCA")
        hbr = _positive(_numbers(head["HBR"], "m"), "HBR")
        pc = _numbers(head["COLLISION_PROBABILITY"], None)
        if not 0 <= pc <= 1:
            raise ValueError("reported probability outside [0,1]")
        objects = []
        for section in sections[1:]:
            f = section["fields"]
            if f["REF_FRAME"] != "ITRF":
                raise ValueError("only explicit ITRF supported")
            if f.get("COV_REF_FRAME", "RTN") != "RTN":
                raise ValueError("only RTN covariance supported")
            state = [
                _numbers(f[k], "km" if i < 3 else "km/s") * 1000
                for i, k in enumerate(STATE_KEYS)
            ]
            c = np.zeros((6, 6))
            for i, axis in enumerate(AXES):
                for j in range(i + 1):
                    unit = "m**2" if i < 3 else ("m**2/s" if j < 3 else "m**2/s**2")
                    c[i, j] = c[j, i] = _numbers(f[f"C{axis}_{AXES[j]}"], unit)
            _covariance(c, 6)
            objects.append(
                {
                    "name": f["OBJECT_NAME"],
                    "designator": f["OBJECT_DESIGNATOR"],
                    "frame": f["REF_FRAME"],
                    "state_itrf_si": state,
                    "covariance_frame": "RTN",
                    "covariance_rtn_si": c.tolist(),
                    "density_sigma": _numbers(f[DCP_SIGMA], None)
                    if DCP_SIGMA in f
                    else None,
                    "sensitivity_position_rtn_m": _numbers(f[DCP_POS], "m", 3)
                    if DCP_POS in f
                    else None,
                    "sensitivity_velocity_rtn_mps": _numbers(f[DCP_VEL], "m/sec", 3)
                    if DCP_VEL in f
                    else None,
                    "source_fields": f,
                    "source_comments": section["comments"],
                }
            )
        result = {
            "schema_version": 1,
            "message_id": head["MESSAGE_ID"],
            "creation_date": head["CREATION_DATE"],
            "tca": head["TCA"],
            "hbr_m": hbr,
            "reported_pc": pc,
            "reported_method": head["COLLISION_PROBABILITY_METHOD"],
            "source_fields": head,
            "source_comments": sections[0]["comments"],
            "objects": objects,
        }
        # Also validate states and optional DCP fields at the parse boundary.
        _prepared(result, np.eye(3))
        return result
    except KeyError as exc:
        raise ValueError(f"missing required CDM field {exc.args[0]}") from exc


def _prepared(cdm, orientation):
    q = np.asarray(orientation, dtype=float)
    if (
        q.shape != (3, 3)
        or not np.isfinite(q).all()
        or not np.allclose(q.T @ q, np.eye(3), atol=1e-12, rtol=0)
        or not np.isclose(np.linalg.det(q), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("orientation must be a proper orthonormal rotation")
    _positive(cdm["hbr_m"], "HBR")
    if len(cdm["objects"]) != 2:
        raise ValueError("two objects required")
    objects = []
    for obj in cdm["objects"]:
        if obj["frame"] != "ITRF" or obj["covariance_frame"] != "RTN":
            raise ValueError("unsupported frame")
        y = _vector(obj["state_itrf_si"], 6)
        r = q @ y[:3]
        v = q @ (y[3:] + np.cross([0, 0, EARTH_OMEGA_RAD_S], y[:3]))
        radial = r / _positive(np.linalg.norm(r), "position norm")
        h = np.cross(r, v)
        normal = h / _positive(np.linalg.norm(h), "angular momentum norm")
        basis = np.column_stack((radial, np.cross(normal, radial), normal))
        cov = _covariance(obj["covariance_rtn_si"], 6)
        transform = np.zeros((6, 6))
        transform[:3, :3] = transform[3:, 3:] = basis
        density = obj.get("density_sigma")
        if density is not None:
            _positive(density, "density sigma")
        g = obj.get("sensitivity_position_rtn_m")
        if g is not None:
            g = basis @ _vector(g, 3)
        hv = obj.get("sensitivity_velocity_rtn_mps")
        if hv is not None:
            _vector(hv, 3)
        objects.append(
            {
                "name": obj["name"],
                "state_common_si": np.r_[r, v].tolist(),
                "basis_rtn_to_common": basis.tolist(),
                "covariance_common_si": _symmetric(
                    transform @ cov @ transform.T
                ).tolist(),
                "density_sigma": density,
                "sensitivity_position_common_m": None if g is None else g.tolist(),
            }
        )
    return objects


def evaluate_cdm(cdm, orientation=None):
    """Return baseline and optional NASA N-18 density sensitivity separately.

    Optional orientation rotates *everything* after Earth velocity correction;
    useful for invariant tests, not an EOP/J2000 conversion API.
    """
    objects = _prepared(cdm, np.eye(3) if orientation is None else orientation)
    a, b = [np.asarray(o["state_common_si"]) for o in objects]
    miss, velocity = b[:3] - a[:3], b[3:] - a[3:]
    speed = _positive(np.linalg.norm(velocity), "relative velocity")
    if speed < 1:
        raise ValueError("near-zero relative speed: 2D method inapplicable")
    normal = velocity / speed
    helper = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    plane = np.column_stack((u, np.cross(normal, u)))
    mean = plane.T @ miss
    ca, cb = [np.asarray(o["covariance_common_si"]) for o in objects]
    common = ca[:3, :3] + cb[:3, :3]
    hbr = cdm["hbr_m"]

    def probability(cov):
        cov = _covariance(cov, 3)
        projected = _covariance(_symmetric(plane.T @ cov @ plane), 2)
        duration = (hbr + 3 * np.sqrt(np.linalg.eigvalsh(projected).max())) / speed
        dynamical = np.sqrt(
            min(np.linalg.norm(a[:3]), np.linalg.norm(b[:3])) ** 3 / 3.986004418e14
        )
        longitudinal = abs(miss @ normal)
        if duration + longitudinal / speed > 0.01 * dynamical:
            raise ValueError("short linear TCA encounter conditions not satisfied")
        return dict(
            disk_probability(mean, projected, hbr, order=96),
            covariance_common_m2=cov.tolist(),
            covariance_plane_m2=projected.tolist(),
            eigenvalues_m2=np.linalg.eigvalsh(cov).tolist(),
            determinant_m6=float(np.linalg.det(cov)),
            sigma_m=np.sqrt(np.linalg.eigvalsh(cov)).tolist(),
            encounter_duration_3sigma_s=float(duration),
            longitudinal_miss_m=float(longitudinal),
            linear_closest_approach_offset_s=float(-miss @ velocity / speed**2),
        )

    baseline = dict(
        probability(common),
        status="evaluated_independent_covariances",
        assumption="NASA N-17: statistically independent object errors",
    )
    sensitivity = {"status": "not_evaluated_missing_dcp", "pc": None}
    if all(
        o["density_sigma"] is not None
        and o["sensitivity_position_common_m"] is not None
        for o in objects
    ):
        gp, gs = [np.asarray(o["sensitivity_position_common_m"]) for o in objects]
        cross = (
            objects[0]["density_sigma"]
            * objects[1]["density_sigma"]
            * (np.outer(gp, gs) + np.outer(gs, gp))
        )
        sensitivity = dict(
            probability(common - cross),
            status="evaluated_sourced_density_hypothesis",
            assumption="NASA N-18 shared global density component; not empirically calibrated here",
            cross_component_m2=cross.tolist(),
        )
    basis_a = np.array(objects[0]["basis_rtn_to_common"])
    return {
        "schema_version": 1,
        "validation_status": "evaluated_bounded_tca",
        "message_id": cdm["message_id"],
        "tca": cdm["tca"],
        "hbr_m": hbr,
        "reported_pc": cdm["reported_pc"],
        "reported_method": cdm["reported_method"],
        "baseline": baseline,
        "correlation_sensitivity": sensitivity,
        "objects": objects,
        "combined_state_covariance_common_si": (ca + cb).tolist(),
        "relative_position_common_m": miss.tolist(),
        "relative_velocity_common_mps": velocity.tolist(),
        "relative_speed_mps": float(speed),
        "miss_distance_m": float(np.linalg.norm(miss)),
        "relative_position_primary_rtn_m": (basis_a.T @ miss).tolist(),
        "relative_velocity_primary_rtn_mps": (basis_a.T @ velocity).tolist(),
        "encounter_basis_common": plane.tolist(),
        "mean_plane_m": mean.tolist(),
        "earth_omega_rad_s": EARTH_OMEGA_RAD_S,
        "model_conditions": {
            "gaussian_marginal_position_errors": True,
            "short_linear_encounter": True,
            "disk_hbr_summed_spheres": True,
            "covariances_supplied_at_tca": True,
            "common_frame": "instantaneous inertial orientation aligned with ITRF at TCA",
            "earth_rotation": "nominal z-axis spin; no EOP/polar-motion/LOD rates",
            "propagator": None,
            "source_force_models_are_input_metadata_only": True,
            "flight_calibration": False,
            "flight_authority": False,
            "operational_replay": False,
            "missing": [
                "pre-TCA operational ephemeris",
                "planned burn",
                "mission constraints",
                "other-object catalog",
            ],
        },
    }
