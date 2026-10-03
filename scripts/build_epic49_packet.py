#!/usr/bin/env python3
"""Freeze once from a read-only archive, then recompute from compact inputs only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import scipy
import sgp4
from sgp4.api import Satrec, WGS72

from services import demo_numerics as n
from scripts import epic49_reference as ref

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/epic49"
ARCHIVE_HASH = "c21a64976cb01a32ffeb3d3784e8fc7ef087d11c2ce537c4acfb6f1a65717c01"
EPOCH = "2019-09-02T10:02:00Z"
CUTOFF = 19245 + 10 / 24


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def record(line1, line2, number):
    raw = line1 + line2
    return {
        "id": line1[2:7].strip(),
        "tle": [line1.rstrip("\r\n"), line2.rstrip("\r\n")],
        "line_numbers": [number, number + 1],
        "raw_text": raw,
        "raw_sha256": hashlib.sha256(raw.encode("ascii")).hexdigest(),
        "epoch_tle": float(line1[18:32]),
    }


def body(identifier, state, radius, covariance=None, kind="constructed"):
    return {
        "id": identifier,
        "kind": kind,
        "epoch_utc": EPOCH,
        "frame": "TEME_frozen_axes",
        "state_si": np.asarray(state).tolist(),
        "position_m": np.asarray(state)[:3].tolist(),
        "velocity_mps": np.asarray(state)[3:].tolist(),
        "radius_m": radius,
        "covariance_ric_m2": covariance,
        "covariance_assumption": "independent position covariance at every TCA, not transported or calibrated"
        if covariance
        else "missing",
    }


def freeze(archive, destination):
    """One bounded streaming pass; no archive writes, no full-file RAM load."""
    sha = hashlib.sha256()
    latest = {}
    pinned = {}
    alternate = None
    pending = None
    counters = dict(
        lines=0,
        matching_pairs=0,
        excluded_epoch_records=0,
        malformed_epoch_records=0,
        nonmatching_or_unpaired_line2=0,
    )
    for number, raw in enumerate(archive.open("rb"), 1):
        sha.update(raw)
        counters["lines"] = number
        line = raw.decode("ascii")
        if line.startswith("1 "):
            pending = (line, number)
        elif line.startswith("2 "):
            if (
                pending is None
                or pending[0][2:7] != line[2:7]
                or pending[1] + 1 != number
            ):
                counters["nonmatching_or_unpaired_line2"] += 1
                pending = None
                continue
            l1, num = pending
            pending = None
            counters["matching_pairs"] += 1
            rec = record(l1, line, num)
            if num in (10513585, 10499818):
                pinned[rec["id"]] = rec
            try:
                epoch = float(l1[18:32])
            except ValueError:
                counters["malformed_epoch_records"] += 1
                continue
            if rec["id"] == "44278" and abs(epoch - 19245.41943374) < 1e-8:
                alternate = rec
            if not CUTOFF - 3 <= epoch <= CUTOFF:
                counters["excluded_epoch_records"] += 1
                continue
            old = latest.get(rec["id"])
            if old is None or (epoch, num) > (old["epoch_tle"], old["line_numbers"][0]):
                latest[rec["id"]] = rec
        else:
            pending = None
    if sha.hexdigest() != ARCHIVE_HASH:
        raise ValueError("archive hash mismatch")
    if set(pinned) != {"43600", "44278"} or alternate is None:
        raise ValueError("missing pinned inputs")
    eligible = []
    excluded_radial = excluded_primary = propagation_errors = 0
    primary_path = n.tle_trajectory(pinned["43600"], EPOCH)
    for identifier in sorted(latest, key=int):
        rec = latest[identifier]
        if identifier in pinned:
            excluded_primary += 1
            continue
        sat = Satrec.twoline2rv(*rec["tle"], WGS72)
        # sat.no_kozai is rad/min: convert explicitly to rad/s, mean Kepler radial envelope.
        mean_motion_rad_s = sat.no_kozai / 60
        semi_major_m = (n.MU / mean_motion_rad_s**2) ** (1 / 3)
        perigee = semi_major_m * (1 - sat.ecco) - n.EARTH_RADIUS_M
        apogee = semi_major_m * (1 + sat.ecco) - n.EARTH_RADIUS_M
        if not (100000 <= perigee <= 345000 and apogee >= 302000 and apogee <= 2000000):
            excluded_radial += 1
            continue
        try:
            path = n.tle_trajectory(rec, EPOCH)
            geometry = n.closest_approach(primary_path, path, 0, 7200, grid_s=60)
            state = path(0)
            if np.linalg.norm(state[:3]) <= n.EARTH_RADIUS_M:
                raise ValueError("subsurface")
        except ValueError:
            propagation_errors += 1
            continue
        rec.update(
            perigee_m=float(perigee),
            apogee_m=float(apogee),
            relevance_minimum=geometry,
            state_at_common_epoch_si=state.tolist(),
        )
        eligible.append(rec)
    # Geometric rank first, catalog ID tie break; no first-N by ID shortcut.
    eligible.sort(key=lambda r: (r["relevance_minimum"]["distance_m"], int(r["id"])))
    selected = eligible[:32]
    catalog = {
        "selection": "latest epoch <= 2019-09-02T10:00Z, age <=3 days; mean radial envelope overlaps 302-345km, perigee>=100km, apogee<=2000km; rank by refined SGP4 minimum to Aeolus over 10:02-12:02Z; nearest 32",
        "publication_time_verified": False,
        "epoch_policy": "retrospective_epoch_cutoff",
        "source_sha256": sha.hexdigest(),
        "stream_counts": counters,
        "latest_age_eligible_ids": len(latest),
        "excluded_primary_secondary_ids": excluded_primary,
        "excluded_radial_ids": excluded_radial,
        "excluded_propagation_ids": propagation_errors,
        "geometrically_eligible_ids": len(eligible),
        "truncated_ids": max(0, len(eligible) - 32),
        "selected_ids": [r["id"] for r in selected],
        "objects": selected,
        "subset_sha256": hashlib.sha256(
            "".join(r["raw_text"] for r in selected).encode("ascii")
        ).hexdigest(),
    }
    historical = {
        "primary": pinned["43600"],
        "secondary": pinned["44278"],
        "alternate_secondary": alternate,
        "search_epoch_utc": "2019-09-02T11:02:00Z",
        "publication_time_verified": False,
        "replay_kind": "retrospective_epoch_cutoff",
        "selection_cutoff_utc": "2019-09-02T10:00:00Z",
        "source_sha256": sha.hexdigest(),
    }
    options = [
        {"id": name, "burn_time_s": 0, "dv_ric_mps": dv}
        for name, dv in [
            ("no-burn", [0, 0, 0]),
            ("along-plus", [0, 0.3, 0]),
            ("along-minus", [0, -0.3, 0]),
            ("cross-plus", [0, 0, 0.3]),
            ("cross-minus", [0, 0, -0.3]),
        ]
    ]
    thresholds = {
        "screen_m": 50000,
        "alert_pc": 1e-5,
        "reject_pc": 1e-4,
        "max_dv_mps": 0.5,
    }
    cov = np.diag([100**2, 200**2, 100**2]).tolist()
    radius = 7000000.0
    speed = np.sqrt(n.MU / radius)
    seed_a = np.array([radius, 0, 0, 0, speed, 0])
    seed_b = np.array([radius + 100, 0, 0, 0, 0, np.sqrt(n.MU / (radius + 100))])
    a0 = ref.trajectory(seed_a, -3600)(-3600)
    b0 = ref.trajectory(seed_b, -3600)(-3600)
    # Declared synthetic crossing constructed from an actual prograde candidate at t=5400.
    candidate = ref.maneuver(a0, 7200, 0, np.array([0, 0.3, 0]))
    intersection = candidate(5400)
    radial = intersection[:3] / np.linalg.norm(intersection[:3])
    transverse = np.cross(radial, intersection[3:])
    transverse /= np.linalg.norm(transverse)
    hazard_seed = np.r_[
        intersection[:3], transverse * np.sqrt(n.MU / np.linalg.norm(intersection[:3]))
    ]
    hazard0 = ref.trajectory(hazard_seed, -5400)(-5400)
    benchmark = {
        "kind": "CONSTRUCTED_OPERATIONAL_BENCHMARK",
        "epoch_utc": EPOCH,
        "frame": "TEME_frozen_axes",
        "horizon_s": 7200,
        "assumed_scenario": True,
        "primary": body("constructed-primary", a0, 4, cov),
        "secondary": body("constructed-secondary", b0, 4, cov),
        "catalog": [body("synthetic-crossing", hazard0, 1, cov)],
        "options": options,
        "thresholds": thresholds,
        "construction": {
            "method": "DOP853 backward from declared crossing seeds, no historical TLE alteration",
            "encounter_time_s": 3600,
            "primary_encounter_seed_si": seed_a.tolist(),
            "secondary_encounter_seed_si": seed_b.tolist(),
            "hazard_intersection_time_s": 5400,
            "hazard_seed_si": hazard_seed.tolist(),
            "hazard_label": "synthetic fixture construction from along-plus trajectory; not discovered debris",
            "covariance_label": "all sigmas assumed, not empirically calibrated",
        },
    }
    real_bodies = [
        body(
            r["id"],
            r["state_at_common_epoch_si"],
            1,
            kind="real_TLE_initial_state_assumed_radius",
        )
        for r in selected
    ]
    historical_dynamics = {
        "kind": "HISTORICAL_INITIAL_STATES_MODEL_EXPERIMENT",
        "epoch_utc": EPOCH,
        "frame": "TEME_frozen_axes",
        "horizon_s": 7200,
        "assumed_scenario": False,
        "primary": body(
            "43600", primary_path(0), 4, kind="real_TLE_initial_state_assumed_radius"
        ),
        "secondary": body(
            "44278",
            n.tle_trajectory(pinned["44278"], EPOCH)(0),
            4,
            kind="real_TLE_initial_state_assumed_radius",
        ),
        "catalog": real_bodies,
        "options": options,
        "thresholds": thresholds,
    }
    data = {
        "schema_version": 1,
        "historical": historical,
        "real_catalog": catalog,
        "benchmark": benchmark,
        "historical_dynamics": historical_dynamics,
        "models": {
            "mu_m3_s2": n.MU,
            "earth_radius_m": n.EARTH_RADIUS_M,
            "J2": n.J2,
            "state_model": "frozen TEME-axis two-body/J2; no drag or axis precession",
            "production_integrator": "RK4 5s + cubic Hermite dense output",
            "reference_integrator": "DOP853 rtol 2e-13, atol 1e-7, max_step20s",
            "covariance": "independent RIC position assumptions at each TCA; no OD/transport",
        },
    }
    write(destination, data)
    print(
        json.dumps(
            {
                "inputs_sha256": digest(data),
                "catalog_counts": {
                    k: v
                    for k, v in catalog.items()
                    if k not in ("objects", "selection")
                },
            }
        )
    )


def build(data):
    return {
        "schema_version": 1,
        "inputs_sha256": digest(data),
        "historical_geometry": n.historical_geometry(data["historical"]),
        "alternate_geometry": n.historical_geometry(data["historical"], alternate=True),
        "benchmark": n.evaluate_case(data["benchmark"]),
        "historical_dynamics": n.evaluate_case(data["historical_dynamics"]),
        "limitations": [
            "Retrospective TLE epochs do not verify contemporaneous publication availability",
            "Historical Pc not evaluated: covariance missing; real radii unverified",
            "Constructed catalog contains one synthetic crossing trajectory, separate from real catalog",
            "Finite dated subset and two-hour horizon provide no global safety assurance",
            "Assumed TCA covariance; no calibration, dynamics transport, drag or flight accuracy",
            "All recommendations are drafts; no spacecraft commands",
        ],
    }


def validate(data, results):
    h = data["historical"]
    checks = []
    historical_reference = ref.minimum(
        ref.tle_path(h["primary"], h["search_epoch_utc"]),
        ref.tle_path(h["secondary"], h["search_epoch_utc"]),
        120,
        grid=1,
    )
    checks.append(
        {
            "kind": "historical_SGP4_independent_minimum",
            "reference": historical_reference,
            "distance_error_m": abs(
                historical_reference["distance_m"]
                - results["historical_geometry"]["distance_m"]
            ),
            "time_error_s": abs(
                historical_reference["time_s"]
                - results["historical_geometry"]["time_s"]
            ),
        }
    )
    if checks[-1]["distance_error_m"] > 1 or checks[-1]["time_error_s"] > 0.01:
        raise AssertionError(checks[-1])
    max_distance = max_time = max_pc = max_position = 0.0
    # Independently integrate EVERY supplied body and candidate, including real secondaries.
    for key in ("benchmark", "historical_dynamics"):
        case = data[key]
        horizon = case["horizon_s"]
        bodies = [case["secondary"]] + case["catalog"]
        references = {b["id"]: ref.trajectory(b["state_si"], horizon) for b in bodies}
        for body_ in [case["primary"]] + bodies:
            coarse = n.trajectory(body_["state_si"], horizon, 10)
            standard = n.trajectory(body_["state_si"], horizon, 5)
            fine = n.trajectory(body_["state_si"], horizon, 2.5)
            reference = ref.trajectory(body_["state_si"], horizon)
            times = np.linspace(0, horizon, 49)
            errors = [
                max(
                    float(np.linalg.norm(path(t)[:3] - reference(t)[:3])) for t in times
                )
                for path in (coarse, standard, fine)
            ]
            max_position = max(max_position, errors[1])
            if errors[1] > 1 or errors[2] > errors[1] + 1e-5:
                raise AssertionError(errors)
            checks.append(
                {
                    "kind": "integrator_convergence",
                    "case": key,
                    "object_id": body_["id"],
                    "max_position_errors_m_at_steps_10_5_2p5": errors,
                }
            )
        for option, actual in zip(case["options"], results[key]["options"]):
            a = ref.maneuver(
                case["primary"]["state_si"],
                horizon,
                option["burn_time_s"],
                np.array(option["dv_ric_mps"]),
            )
            actual_path = n.maneuver_trajectory(
                case["primary"]["state_si"],
                horizon,
                option["burn_time_s"],
                option["dv_ric_mps"],
            )
            position_error = max(
                float(np.linalg.norm(actual_path(t)[:3] - a(t)[:3]))
                for t in np.linspace(0, horizon, 49)
            )
            max_position = max(max_position, position_error)
            if position_error > 1:
                raise AssertionError((key, option["id"], position_error))
            checks.append(
                {
                    "kind": "burned_path_position",
                    "case": key,
                    "option_id": option["id"],
                    "max_position_error_m": position_error,
                }
            )
            for body_, encounter in zip(
                bodies, [actual["primary_encounter"]] + actual["secondary_encounters"]
            ):
                b = references[body_["id"]]
                best = ref.minimum(a, b, horizon)
                distance_error = abs(best["distance_m"] - encounter["distance_m"])
                time_error = abs(best["time_s"] - encounter["time_s"])
                max_distance = max(max_distance, distance_error)
                max_time = max(max_time, time_error)
                if distance_error > 1 or time_error > 0.01:
                    raise AssertionError(
                        (key, option["id"], body_["id"], distance_error, time_error)
                    )
                check = {
                    "kind": "candidate_encounter",
                    "case": key,
                    "option_id": option["id"],
                    "object_id": body_["id"],
                    "reference": best,
                    "distance_error_m": distance_error,
                    "time_error_s": time_error,
                }
                if encounter["probability"]["pc"] is not None:
                    t = best["time_s"]
                    probability = ref.probability(
                        a(t),
                        b(t),
                        np.array(case["primary"]["covariance_ric_m2"]),
                        np.array(body_["covariance_ric_m2"]),
                        case["primary"]["radius_m"] + body_["radius_m"],
                    )
                    error = abs(probability["pc"] - encounter["probability"]["pc"])
                    tolerance = 1e-12 + 1e-5 * abs(probability["pc"])
                    if error > tolerance:
                        raise AssertionError((error, tolerance))
                    max_pc = max(max_pc, error)
                    check.update(
                        reference_probability=probability,
                        pc_abs_error=error,
                        pc_tolerance=tolerance,
                    )
                    same_state_reference = ref.probability(
                        actual_path(encounter["time_s"]),
                        n.trajectory(body_["state_si"], horizon)(encounter["time_s"]),
                        np.array(case["primary"]["covariance_ric_m2"]),
                        np.array(body_["covariance_ric_m2"]),
                        case["primary"]["radius_m"] + body_["radius_m"],
                    )
                    quadrature_error = abs(
                        same_state_reference["pc"] - encounter["probability"]["pc"]
                    )
                    if quadrature_error > tolerance:
                        raise AssertionError(quadrature_error)
                    check.update(
                        same_state_reference_probability=same_state_reference,
                        isolated_quadrature_abs_error=quadrature_error,
                    )
                checks.append(check)
    sensitivity = []
    baseline = results["benchmark"]
    for modification in ("magnitude", "time", "state", "renamed_reordered"):
        changed = copy.deepcopy(data["benchmark"])
        if modification == "magnitude":
            changed["options"][1]["dv_ric_mps"][1] *= 2
        elif modification == "time":
            changed["options"][1]["burn_time_s"] = 300
        elif modification == "state":
            changed["primary"]["state_si"][0] += 100
        else:
            changed["options"].reverse()
            for option in changed["options"]:
                option["id"] = "renamed_" + option["id"]
        response = n.evaluate_case(changed)
        if modification == "renamed_reordered":
            physical = []
            for option in reversed(response["options"]):
                option["id"] = option["id"].removeprefix("renamed_")
                physical.append(option)
            if physical != baseline["options"]:
                raise AssertionError("IDs changed physical results")
            sensitivity.append(
                {"perturbation": modification, "physical_results_identical": True}
            )
        else:
            before = baseline["options"][1]["primary_encounter"]["distance_m"]
            after = response["options"][1]["primary_encounter"]["distance_m"]
            if abs(after - before) <= 1:
                raise AssertionError("physical perturbation did not change geometry")
            sensitivity.append(
                {
                    "perturbation": modification,
                    "baseline_distance_m": before,
                    "changed_distance_m": after,
                    "distance_change_m": after - before,
                }
            )
    return {
        "inputs_sha256": digest(data),
        "results_sha256": digest(results),
        "tolerances": {
            "distance_m": 1,
            "time_s": 0.01,
            "position_m": 1,
            "pc_abs": 1e-12,
            "pc_rel": 1e-5,
        },
        "measured_max_errors": {
            "distance_m": max_distance,
            "time_s": max_time,
            "position_m": max_position,
            "pc_absolute": max_pc,
        },
        "checks": checks,
        "sensitivity_checks": sensitivity,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sgp4": sgp4.__version__,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-from-archive", type=Path)
    parser.add_argument("--inputs", type=Path, default=FIXTURES / "inputs.json")
    parser.add_argument("--output", type=Path, default=FIXTURES / "computed.json")
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    if args.freeze_from_archive:
        freeze(args.freeze_from_archive, args.inputs)
    else:
        data = json.loads(args.inputs.read_text())
        results = build(data)
        write(args.output, results)
        if args.reference_output:
            write(args.reference_output, validate(data, results))
        print(
            json.dumps(
                {
                    "results_sha256": digest(results),
                    "elapsed_s": time.monotonic() - start,
                    "historical_geometry": results["historical_geometry"],
                }
            )
        )


if __name__ == "__main__":
    main()
