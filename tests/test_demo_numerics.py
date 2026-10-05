"""Behavioral checks for the bounded numerical evidence packet (no product hooks)."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "epic49"


def numerics():
    try:
        return importlib.import_module("services.demo_numerics")
    except ModuleNotFoundError:
        pytest.fail("numerical implementation absent: services.demo_numerics")


def inputs():
    return json.loads((FIXTURE / "inputs.json").read_text())


def test_si_conversion_and_ric_axes():
    n = numerics()
    np.testing.assert_array_equal(
        n.km_state_to_si([7000, 0, 0], [0, 7.5, 0]), [7e6, 0, 0, 0, 7500, 0]
    )
    state = np.array([7e6, 0, 0, 0, 7500, 0])
    np.testing.assert_allclose(n.ric_basis(state), np.eye(3), atol=1e-15)
    np.testing.assert_allclose(
        n.apply_impulse(state, [0, 0.3, -0.2])[3:], [0, 7500.3, -0.2]
    )
    turned = np.array([0, 7e6, 0, -7500, 0, 0])
    np.testing.assert_allclose(n.ric_basis(turned), [[0, -1, 0], [1, 0, 0], [0, 0, 1]])


def test_covariance_missing_invalid_and_low_speed_abstain():
    n = numerics()
    a = np.array([7e6, 0, 0, 0, 7500, 0])
    b = np.array([7e6 + 10, 0, 0, 0, 0, 7500])
    cov = np.diag([100.0, 400.0, 900.0])
    assert n.encounter_probability(a, b, None, cov, 8)["pc"] is None
    for bad in (np.diag([-1, 2, 3]), np.zeros((3, 3)), np.full((3, 3), np.nan)):
        assert (
            n.encounter_probability(a, b, bad, cov, 8)["status"] == "invalid_covariance"
        )
    assert (
        n.encounter_probability(a, a, None, None, 8)["status"] == "low_relative_speed"
    )
    assert n.encounter_probability(a, b, cov, cov, 8, assumed=False)["pc"] is None
    far = b.copy()
    far[0] += 1e5
    assert (
        n.encounter_probability(a, far, cov, cov, 8)["status"]
        == "outside_screened_volume"
    )


def test_centered_isotropic_probability_and_rotation():
    n = numerics()
    a = np.array([7e6, 0, 0, 0, 7500, 0])
    b = np.array([7e6, 0, 0, 0, 0, 7500])
    cov = np.eye(3) * 10000
    result = n.encounter_probability(a, b, cov, cov, 8)
    assert result["pc"] == pytest.approx(-np.expm1(-(8**2) / 40000), rel=1e-10)
    angle = 0.7
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    aa = np.r_[rot @ a[:3], rot @ a[3:]]
    bb = np.r_[rot @ b[:3], rot @ b[3:]]
    assert n.encounter_probability(aa, bb, cov, cov, 8)["pc"] == pytest.approx(
        result["pc"], rel=1e-12
    )


def test_continuous_minimum_catches_between_sample_event():
    n = numerics()

    def a(t):
        return np.array([0, 0, 0, 0, 0, 0])

    def b(t):
        return np.array([(t - 41.662) * 14403.873, 2592.261, 0, 14403.873, 0, 0])

    result = n.closest_approach(a, b, 0, 120, grid_s=30)
    assert result["time_s"] == pytest.approx(41.662, abs=1e-6)
    assert result["distance_m"] == pytest.approx(2592.261, abs=1e-6)


def test_pinned_archive_geometry():
    n = numerics()
    data = inputs()["historical"]
    result = n.historical_geometry(data)
    assert result["distance_m"] == pytest.approx(2592.261, abs=1)
    assert result["time_s"] == pytest.approx(41.662, abs=0.01)
    assert result["relative_speed_mps"] == pytest.approx(14403.873, abs=0.01)
    assert data["publication_time_verified"] is False
    assert data["primary"]["line_numbers"] == [10513585, 10513586]
    assert data["secondary"]["line_numbers"] == [10499818, 10499819]


def test_integrator_independent_reference_and_convergence():
    n = numerics()
    from scripts.epic49_reference import trajectory

    state = np.array(inputs()["benchmark"]["primary"]["state_si"])
    reference = trajectory(state, 7200)
    errors = []
    for step in (10, 5, 2.5):
        actual = n.trajectory(state, 7200, step_s=step)
        errors.append(np.linalg.norm(actual(7200)[:3] - reference(7200)[:3]))
    assert errors[0] < 1
    assert errors[2] < errors[1] < errors[0]


def test_candidate_physics_invariant_to_names_and_order_and_sensitive():
    n = numerics()
    case = inputs()["benchmark"]
    base = n.evaluate_case(case)
    changed = copy.deepcopy(case)
    changed["options"].reverse()
    for o in changed["options"]:
        o["id"] = "renamed_" + o["id"]
    reordered = n.evaluate_case(changed)
    for old, new in zip(base["options"], reversed(reordered["options"])):
        assert old["primary_encounter"] == new["primary_encounter"]
        assert old["secondary_encounters"] == new["secondary_encounters"]
    for modification in ("magnitude", "time", "state"):
        changed = copy.deepcopy(case)
        if modification == "magnitude":
            changed["options"][1]["dv_ric_mps"][1] *= 2
        if modification == "time":
            changed["options"][1]["burn_time_s"] = 300
        if modification == "state":
            changed["primary"]["state_si"][0] += 100
        other = n.evaluate_case(changed)
        assert (
            abs(
                other["options"][1]["primary_encounter"]["distance_m"]
                - base["options"][1]["primary_encounter"]["distance_m"]
            )
            > 1
        )


def test_constructed_hazard_follows_states_and_bounded_scope():
    n = numerics()
    case = inputs()["benchmark"]
    result = n.evaluate_case(case)
    prograde = result["options"][1]
    hazard = next(
        e
        for e in prograde["secondary_encounters"]
        if e["object_id"] == "synthetic-crossing"
    )
    assert hazard["distance_m"] < 1
    assert hazard["probability"]["pc"] > case["thresholds"]["reject_pc"]
    assert prograde["rejected"]
    assert result["screen_scope"]["global_safety_claim"] is False
    changed = copy.deepcopy(case)
    changed["catalog"][-1]["state_si"][0] += 10000
    other = n.evaluate_case(changed)["options"][1]["secondary_encounters"][-1]
    assert other["distance_m"] > 1000


def test_missing_historical_covariance_cannot_be_recommended():
    n = numerics()
    result = n.evaluate_case(inputs()["historical_dynamics"])
    assert all(
        o["evaluation_status"] == "abstain_missing_primary_probability"
        for o in result["options"]
    )
    assert result["screen_scope"]["conjunctions_inside_volume"] == 0


def test_anisotropic_covariance_rotates_each_body_and_sums_radii():
    n = numerics()
    from scripts.epic49_reference import probability

    a = np.array([7e6, 0, 0, 0, 7500, 0.0])
    b = np.array([7e6 + 120, 0, 0, 0, -2500, 7500.0])
    ca = np.diag([100**2, 200**2, 50**2])
    cb = np.diag([150**2, 80**2, 250**2])
    expected = probability(a, b, ca, cb, 4 + 1)["pc"]
    actual = n.encounter_probability(a, b, ca, cb, 4 + 1)["pc"]
    assert actual == pytest.approx(expected, abs=1e-12, rel=1e-8)
    assert actual < n.encounter_probability(a, b, ca, cb, 8)["pc"]
    assert actual != pytest.approx(
        n.encounter_probability(a, b, ca, cb[[0, 2, 1]][:, [0, 2, 1]], 5)["pc"],
        rel=0.01,
    )


def test_tiny_probability_retains_log_and_invalid_states_raise():
    n = numerics()
    tiny = n.disk_probability([10000, 0], np.eye(2) * 100, 5)
    assert tiny["pc"] == 0 and tiny["underflow"]
    # Gaussian disk-density upper bound; finite log identifies numerical underflow.
    bound = np.log(25 / 200) - (10000 - 5) ** 2 / 200
    assert np.isfinite(tiny["log_pc"]) and tiny["log_pc"] < bound
    with pytest.raises(ValueError):
        n.trajectory([7e6, 0, 0, 0, np.nan, 0], 10)
    with pytest.raises(ValueError):
        n.trajectory([7000, 0, 0, 0, 7.5, 0], 10)
    with pytest.raises(ValueError):
        n.ric_basis([7e6, 0, 0, 1, 0, 0])


def test_tca_grid_convergence_and_inputs_have_exact_hashes():
    n = numerics()
    data = inputs()
    import hashlib

    catalog = data["real_catalog"]
    raw = "".join(r["raw_text"] for r in catalog["objects"])
    assert hashlib.sha256(raw.encode("ascii")).hexdigest() == catalog["subset_sha256"]
    assert (
        catalog["geometrically_eligible_ids"]
        == len(catalog["objects"]) + catalog["truncated_ids"]
    )
    for rec in [
        data["historical"]["primary"],
        data["historical"]["secondary"],
    ] + catalog["objects"]:
        assert rec["tle"][0][2:7] == rec["tle"][1][2:7]
        assert (
            hashlib.sha256(rec["raw_text"].encode("ascii")).hexdigest()
            == rec["raw_sha256"]
        )
    case = data["benchmark"]
    a = n.trajectory(case["primary"]["state_si"], 7200)
    b = n.trajectory(case["secondary"]["state_si"], 7200)
    results = [n.closest_approach(a, b, 0, 7200, grid_s=g) for g in (60, 30, 10)]
    assert (
        max(r["distance_m"] for r in results) - min(r["distance_m"] for r in results)
        < 0.001
    )
    assert (
        max(r["time_s"] for r in results) - min(r["time_s"] for r in results) < 0.0001
    )


def test_alert_is_distinct_from_rejection_and_null_burn_matches_baseline():
    n = numerics()
    case = copy.deepcopy(inputs()["benchmark"])
    case["options"] = case["options"][:1]
    case["thresholds"]["reject_pc"] = 0.01
    result = n.evaluate_case(case)["options"][0]
    assert result["primary_encounter"]["alert"]
    assert not result["primary_encounter"]["rejected"]
    assert not result["rejected"]
    state = case["primary"]["state_si"]
    base = n.trajectory(state, 7200)
    null = n.maneuver_trajectory(state, 7200, 0, [0, 0, 0])
    np.testing.assert_array_equal(base(1234.5), null(1234.5))


def test_mixed_dates_and_invalid_size_abstain_by_rejecting_input():
    n = numerics()
    for modification in ("epoch", "frame", "radius"):
        case = copy.deepcopy(inputs()["benchmark"])
        if modification == "epoch":
            case["catalog"][0]["epoch_utc"] = "2026-04-12T00:00:00Z"
        if modification == "frame":
            case["catalog"][0]["frame"] = "ITRF"
        if modification == "radius":
            case["catalog"][0]["radius_m"] = float("nan")
        with pytest.raises(ValueError):
            n.evaluate_case(case)
