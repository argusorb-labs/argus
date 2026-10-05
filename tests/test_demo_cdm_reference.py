"""Released data reconstruction: catch unit/frame, own-RTN and covariance errors."""

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures/epic49_round2"


def module():
    try:
        return importlib.import_module("services.demo_cdm_reference")
    except ModuleNotFoundError:
        pytest.fail("CDM reference implementation absent")


def parsed():
    return module().parse_cdm((FIXTURE / "source_cdm.kvn").read_text())


def test_released_fields_and_full_covariances():
    d = parsed()
    assert d["message_id"] == "000044628_conj_000027127_20220313_181420_20220311_225243"
    assert d["tca"] == "2022-03-13T18:14:20.971"
    assert d["hbr_m"] == 4.5
    assert d["reported_pc"] == 1.601e-4
    assert [o["name"] for o in d["objects"]] == ["ICON", "PSLV DEB"]
    a = d["objects"][0]
    assert a["state_itrf_si"][0] == pytest.approx(-5771809.181165332)
    assert a["state_itrf_si"][3] == pytest.approx(-2995.0906313536136)
    assert np.shape(a["covariance_rtn_si"]) == (6, 6)
    assert a["covariance_rtn_si"][5][5] == 6.51833597874e-5
    assert a["density_sigma"] == pytest.approx(0.245519905)
    assert a["sensitivity_position_rtn_m"][1] == pytest.approx(3499.336708808675)


@pytest.mark.parametrize(
    "before,after",
    [
        ("REF_FRAME                                   = ITRF", "REF_FRAME = TEME"),
        ("REF_FRAME                                   = ITRF", ""),
        ("[km/s]", "[m/s]"),
        ("-5.771809181165332120e+03 [km]", "-5.771809181165332120e+03 [m]"),
        ("[m**2]", "[km**2]"),
        (
            "CR_R                                        = 2.536723712995373319e+02 [m**2]",
            "",
        ),
        ("COMMENT HBR = 4.5 [m]", ""),
        ("COMMENT HBR = 4.5 [m]", "COMMENT HBR = 0 [m]"),
        ("-5.771809181165332120e+03", "nan"),
        ("2.536723712995373319e+02", "-2.536723712995373319e+02"),
    ],
)
def test_parser_fails_closed(before, after):
    raw = (FIXTURE / "source_cdm.kvn").read_text()
    assert before in raw
    with pytest.raises(ValueError):
        module().parse_cdm(raw.replace(before, after, 1))


def test_earth_velocity_and_source_covariance_invariants():
    result = module().evaluate_cdm(parsed())
    assert result["baseline"]["pc"] == pytest.approx(1.601e-4, rel=0.0004)
    np.testing.assert_allclose(
        result["baseline"]["eigenvalues_m2"], [2.0282e2, 5.4626e5, 8.3512e6], rtol=6e-5
    )
    assert result["baseline"]["determinant_m6"] == pytest.approx(9.2523e14, rel=6e-5)
    np.testing.assert_allclose(
        result["relative_position_primary_rtn_m"], [-57.3, 917.5, -1707.3], atol=0.051
    )
    np.testing.assert_allclose(
        result["relative_velocity_primary_rtn_mps"],
        [-10.1, -11736, -6332.5],
        atol=0.051,
    )
    # Printed corrected Pc exponent conflicts with its own N-28 covariance.
    # Independent integration below checks the estimate; do not fit to 8.04e-4.
    np.testing.assert_allclose(
        result["correlation_sensitivity"]["sigma_m"],
        [13.980, 189.59, 3396.13],
        rtol=6e-5,
    )
    assert result["correlation_sensitivity"]["determinant_m6"] == pytest.approx(
        8.1020e13, rel=0.0001
    )


def test_inputs_drive_probability_and_reported_value_is_only_target():
    n = module()
    d = parsed()
    base = n.evaluate_cdm(d)["baseline"]["pc"]
    for mode in ("geometry", "covariance", "hbr"):
        other = copy.deepcopy(d)
        if mode == "geometry":
            other["objects"][1]["state_itrf_si"][0] += 30
        elif mode == "covariance":
            other["objects"][1]["covariance_rtn_si"] = (
                np.array(other["objects"][1]["covariance_rtn_si"]) * 1.2
            ).tolist()
        else:
            other["hbr_m"] = 5
        assert abs(n.evaluate_cdm(other)["baseline"]["pc"] / base - 1) > 0.01
    d["reported_pc"] = 0.5
    assert n.evaluate_cdm(d)["baseline"]["pc"] == base


def test_common_rotation_invariance():
    n = module()
    d = parsed()
    q, _ = np.linalg.qr(np.array([[1.0, 2, 3], [4, 2, 1], [1, 1, 5]]))
    a = n.evaluate_cdm(d)
    b = n.evaluate_cdm(d, orientation=q)
    for key in ("baseline", "correlation_sensitivity"):
        assert b[key]["pc"] == pytest.approx(a[key]["pc"], rel=1e-10)
        np.testing.assert_allclose(
            b[key]["eigenvalues_m2"], a[key]["eigenvalues_m2"], rtol=1e-10
        )


@pytest.mark.parametrize(
    "mode",
    [
        "covariance",
        "nonfinite",
        "hbr",
        "velocity",
        "density_missing",
        "density_invalid",
    ],
)
def test_normalized_inputs_cannot_bypass_validation(mode):
    d = parsed()
    if mode == "covariance":
        d["objects"][0]["covariance_rtn_si"] = None
    elif mode == "nonfinite":
        d["objects"][0]["state_itrf_si"][0] = float("inf")
    elif mode == "hbr":
        d["hbr_m"] = -1
    elif mode == "velocity":
        d["objects"][1]["state_itrf_si"] = d["objects"][0]["state_itrf_si"]
    elif mode == "density_missing":
        d["objects"][0]["density_sigma"] = None
        assert (
            module().evaluate_cdm(d)["correlation_sensitivity"]["status"]
            == "not_evaluated_missing_dcp"
        )
        return
    else:
        d["objects"][0]["density_sigma"] = -1
    with pytest.raises(ValueError):
        module().evaluate_cdm(d)


def test_adaptive_reference_independently_reconstructs_geometry_and_pc():
    try:
        ref = importlib.import_module("scripts.epic49_cdm_reference")
    except ModuleNotFoundError:
        pytest.fail("adaptive CDM reference absent")
    d = parsed()
    actual = module().evaluate_cdm(d)
    other = ref.evaluate(d)
    for key in ("baseline", "correlation_sensitivity"):
        assert other[key]["pc"] == pytest.approx(actual[key]["pc"], rel=2e-9)
        assert other[key]["quadrature_error_estimate"] < 1e-12
        np.testing.assert_allclose(
            other[key]["eigenvalues_m2"], actual[key]["eigenvalues_m2"], rtol=1e-9
        )


def test_builder_reports_optional_source_mismatch_and_is_deterministic(tmp_path):
    try:
        builder = importlib.import_module("scripts.build_epic49_cdm_packet")
    except ModuleNotFoundError:
        pytest.fail("CDM evidence packet builder absent")
    builder.build(FIXTURE, tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    builder.build(FIXTURE, tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    import json

    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert (
        receipt["source_comparisons"]["baseline_pc"]["status"]
        == "within_source_rounding"
    )
    corrected = receipt["source_comparisons"]["corrected_pc"]
    assert corrected["target"] == 8.04e-4
    assert corrected["status"] == "incomplete_source_pc_mismatch"
    assert corrected["actual"] == pytest.approx(8.04040496e-5, rel=1e-8)
    assert receipt["independent_agreement"]["baseline"]["absolute_difference"] < 1e-12
    assert receipt["round1_preservation"]["all_base_files_preserved"]
    assert not receipt["round1_preservation"]["all_base_files_unchanged"]
    assert (
        receipt["round1_preservation"]["documentation_changes"]["README.md"]["kind"]
        == "append_only"
    )


@pytest.mark.parametrize("value", [0, float("nan"), float("inf")])
def test_invalid_hbr_rejected(value):
    d = parsed()
    d["hbr_m"] = value
    with pytest.raises(ValueError):
        module().evaluate_cdm(d)


def test_wrong_covariance_frame_and_asymmetry_rejected():
    d = parsed()
    d["objects"][1]["covariance_frame"] = "ITRF"
    with pytest.raises(ValueError):
        module().evaluate_cdm(d)
    d = parsed()
    d["objects"][1]["covariance_rtn_si"][0][1] += 1
    with pytest.raises(ValueError):
        module().evaluate_cdm(d)


def test_density_sensitivity_parameters_drive_corrected_estimate():
    d = parsed()
    base = module().evaluate_cdm(d)
    d["objects"][0]["density_sigma"] *= 0.9
    changed = module().evaluate_cdm(d)
    assert changed["baseline"]["pc"] == base["baseline"]["pc"]
    assert (
        abs(
            changed["correlation_sensitivity"]["pc"]
            / base["correlation_sensitivity"]["pc"]
            - 1
        )
        > 0.01
    )


@pytest.mark.parametrize(
    "change",
    ["append_readme", "rewrite_readme", "science_change", "unrelated_ci_change"],
)
def test_preservation_allows_documentation_append_but_rejects_rewrites(
    tmp_path, monkeypatch, change
):
    import subprocess

    builder = importlib.import_module("scripts.build_epic49_cdm_packet")
    root = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--shared", "--quiet", str(builder.ROOT), str(root)],
        check=True,
    )
    monkeypatch.setattr(builder, "ROOT", root)
    relative = {
        "science_change": "services/demo_numerics.py",
        "unrelated_ci_change": ".github/workflows/ci.yml",
    }.get(change, "README.md")
    target = root / relative
    if change == "rewrite_readme":
        target.write_text("replaced original documentation\n")
    else:
        target.write_bytes(target.read_bytes() + b"\nAdditional text\n")
    if change == "append_readme":
        result = builder.preservation()
        assert result["all_base_files_preserved"]
        assert not result["all_base_files_unchanged"]
    else:
        with pytest.raises(ValueError, match="round-1 files changed"):
            builder.preservation()
