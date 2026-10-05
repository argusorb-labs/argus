"""Deterministically rebuild released-CDM evidence; no service or flight hooks.

Run: python -m scripts.build_epic49_cdm_packet [--output PATH] [--verify-pdf PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import scipy

from scripts.epic49_cdm_reference import evaluate as reference
from services.demo_cdm_reference import evaluate_cdm, parse_cdm

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/epic49_round2"
BASE = "2db2eaaf2bfc8b6081f91a475089525b47a50c07"
ANCHOR = "3f49212"


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def preservation():
    paths = (
        subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT)
        .decode()
        .splitlines()
    )
    mismatches = []
    records = {}
    documentation_changes = {}
    ci_changes = {}
    for path in paths:
        target = ROOT / path
        # Symlinks are outside this additive packet; git diff verifies them.
        if target.is_symlink():
            continue
        old = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
        current = target.read_bytes() if target.is_file() else None
        if current != old:
            # Product knowledge may append README links; the original bytes and
            # every other baseline file remain protected by the historical gate.
            if path == "README.md" and current is not None and current.startswith(old):
                documentation_changes[path] = {
                    "kind": "append_only",
                    "base_sha256": sha(old),
                    "current_sha256": sha(current),
                }
            elif path == ".github/workflows/ci.yml" and current == old.replace(
                b"      - uses: actions/checkout@v4\n\n      - uses: astral-sh/setup-uv@v6",
                b"      - uses: actions/checkout@v4\n        with:\n"
                b"          # Evidence builders verify historical scientific commit anchors.\n"
                b"          fetch-depth: 0\n\n      - uses: astral-sh/setup-uv@v6",
                1,
            ):
                ci_changes[path] = {
                    "kind": "historical_anchor_checkout",
                    "base_sha256": sha(old),
                    "current_sha256": sha(current),
                }
            else:
                mismatches.append(path)
        records[path] = sha(old)
    science_paths = [
        "services/demo_numerics.py",
        "scripts/epic49_reference.py",
        "scripts/build_epic49_packet.py",
        "tests/test_demo_numerics.py",
    ]
    science_paths += sorted(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "tests/fixtures/epic49").iterdir()
        if p.is_file()
    )
    anchor = {}
    for path in science_paths:
        old = subprocess.check_output(["git", "show", f"{ANCHOR}:{path}"], cwd=ROOT)
        anchor[path] = {
            "sha256": sha(old),
            "unchanged": old == (ROOT / path).read_bytes(),
        }
    if mismatches or not all(v["unchanged"] for v in anchor.values()):
        raise ValueError("round-1 files changed")
    return {
        "base": BASE,
        "science_anchor": ANCHOR,
        "all_base_files_unchanged": not (documentation_changes or ci_changes),
        "all_base_files_preserved": True,
        "documentation_changes": documentation_changes,
        "ci_changes": ci_changes,
        "checked_regular_files": len(records),
        "base_file_manifest_sha256": sha(encoded(records)),
        "science_files": anchor,
    }


def build(source=FIXTURE, destination=FIXTURE, pdf=None):
    source, destination = Path(source), Path(destination)
    raw = (source / "source_cdm.kvn").read_bytes()
    provenance = json.loads((source / "provenance.json").read_text())
    if sha(raw) != provenance["extraction"]["kvn_sha256"]:
        raise ValueError("CDM extract hash mismatch")
    if pdf is not None and sha(Path(pdf).read_bytes()) != provenance["sha256"]:
        raise ValueError("PDF source hash mismatch")
    parsed = parse_cdm(raw.decode())
    actual = evaluate_cdm(parsed)
    independent = reference(parsed)
    actual["source_hashes"] = {
        "pdf_sha256": provenance["sha256"],
        "kvn_sha256": sha(raw),
        "provenance_sha256": sha((source / "provenance.json").read_bytes()),
    }
    agreement = {}
    for key in ("baseline", "correlation_sensitivity"):
        delta = abs(actual[key]["pc"] - independent[key]["pc"])
        relative = delta / independent[key]["pc"]
        if relative > 2e-9:
            raise ValueError("independent reference disagrees")
        agreement[key] = {
            "absolute_difference": delta,
            "relative_difference": relative,
            "relative_tolerance": 2e-9,
            "status": "agrees",
            "adaptive_error_estimate": independent[key]["quadrature_error_estimate"],
            "adaptive_inner_error_bound_estimate": independent[key][
                "inner_error_bound_estimate"
            ],
        }
    comparisons = {}

    def compare(name, actual_value, target, tolerance, mismatch="source_mismatch"):
        error = abs(np.asarray(actual_value) / np.asarray(target) - 1)
        ok = bool(np.all(error <= tolerance))
        comparisons[name] = {
            "actual": actual_value,
            "target": target,
            "max_relative_difference": float(np.max(error)),
            "relative_tolerance": tolerance,
            "status": "within_source_rounding" if ok else mismatch,
        }
        return ok

    # Independent source targets, never used in the probability calculation.
    required = [
        compare("baseline_pc", actual["baseline"]["pc"], parsed["reported_pc"], 4e-4),
        compare(
            "baseline_eigenvalues_m2",
            actual["baseline"]["eigenvalues_m2"],
            [202.82, 546260, 8351200],
            6e-5,
        ),
        compare(
            "baseline_determinant_m6",
            actual["baseline"]["determinant_m6"],
            9.2523e14,
            6e-5,
        ),
        compare(
            "corrected_determinant_m6",
            actual["correlation_sensitivity"]["determinant_m6"],
            8.1020e13,
            1e-4,
        ),
        compare(
            "corrected_sigma_m",
            actual["correlation_sensitivity"]["sigma_m"],
            [13.980, 189.59, 3396.13],
            6e-5,
        ),
    ]
    if not all(required):
        raise ValueError("source covariance/baseline comparison failed")
    compare(
        "corrected_pc",
        actual["correlation_sensitivity"]["pc"],
        8.04e-4,
        1e-3,
        "incomplete_source_pc_mismatch",
    )
    code_paths = [
        "services/demo_cdm_reference.py",
        "services/demo_numerics.py",
        "scripts/epic49_cdm_reference.py",
        "scripts/build_epic49_cdm_packet.py",
    ]
    artifacts = {
        "source_cdm.json": parsed,
        "computed.json": actual,
        "independent.json": independent,
    }
    receipt = {
        "schema_version": 1,
        "baseline_verification": "source_and_independent_integral_agree",
        "optional_corrected_source_pc_verification": comparisons["corrected_pc"][
            "status"
        ],
        "source_comparisons": comparisons,
        "independent_agreement": agreement,
        "source_arithmetic_conflict": {
            "pdf_page_1_based": 94,
            "printed_corrected_pc": 8.04e-4,
            "printed_baseline_pc": 1.60e-4,
            "printed_values_ratio": 8.04e-4 / 1.60e-4,
            "computed_corrected_to_baseline_ratio": actual["correlation_sensitivity"][
                "pc"
            ]
            / actual["baseline"]["pc"],
            "diagnosis": "N-18, DCP data, N-28 invariants and two integrals imply about 8.04e-5; printed exponent conflicts with reduction prose. Possible source typo is an inference, not confirmed erratum.",
            "resolution": "preserve source target; optional published-Pc reproduction incomplete; no fitting or data substitution",
        },
        "algorithms": {
            "production": "96x96 polar Gauss-Legendre; accepted pure disk_probability",
            "reference": "independent RTN construction/encounter axes; nested adaptive Cartesian scipy quad",
            "shared_assumptions": "same source interpretation, nominal Earth spin, Gaussian short encounter and N-18 density hypothesis",
        },
        "source_hashes": actual["source_hashes"],
        "code_sha256": {p: sha((ROOT / p).read_bytes()) for p in code_paths},
        "artifact_sha256": {
            name: sha(encoded(value)) for name, value in artifacts.items()
        },
        "round1_preservation": preservation(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "model_conditions": actual["model_conditions"],
        "rebuild_command": "python -m scripts.build_epic49_cdm_packet",
    }
    artifacts["receipt.json"] = receipt
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        (destination / name).write_bytes(encoded(value))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIXTURE)
    parser.add_argument("--verify-pdf", type=Path)
    args = parser.parse_args()
    result = build(destination=args.output, pdf=args.verify_pdf)
    print(
        json.dumps(
            {
                "baseline": result["baseline_verification"],
                "optional_corrected": result[
                    "optional_corrected_source_pc_verification"
                ],
                "artifact_sha256": result["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
