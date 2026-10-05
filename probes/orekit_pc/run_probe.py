"""Real Orekit 13.1 projected-2D Pc probe; no simulated or cached result."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--argonavis-root", type=Path, required=True)
parser.add_argument("--deps", type=Path, default=Path("/private/tmp/argus-orekit-deps"))
parser.add_argument(
    "--java-bin", type=Path, default=Path("/opt/homebrew/opt/openjdk/bin")
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.argonavis_root.resolve()
deps = args.deps.resolve()
here = Path(__file__).resolve().parent
lock = json.loads((here / "dependencies.json").read_text())
for item in lock:
    assert (
        hashlib.sha256((deps / item["filename"]).read_bytes()).hexdigest()
        == item["sha256"]
    ), item["filename"]
sys.path.insert(0, str(root / "src"))
from argonavis.services.epic49_cdm_reference import evaluate_cdm, parse_cdm
from argonavis.services.epic49_numerics import disk_probability

source = root / "src/argonavis/data/epic49_round2/source_cdm.kvn"
cdm = parse_cdm(source.read_text())
result = evaluate_cdm(cdm)
mean = np.asarray(result["mean_plane_m"])
cov = np.asarray(result["baseline"]["covariance_plane_m2"])
eigen, vectors = np.linalg.eigh(cov)
if np.linalg.det(vectors) < 0:
    vectors[:, 1] *= -1
rotated = vectors.T @ mean
sigmas = np.sqrt(eigen)
assert np.allclose(vectors.T @ cov @ vectors, np.diag(eigen), rtol=1e-12, atol=1e-8)
classes = Path("/private/tmp/argus-orekit-probe-classes")
classes.mkdir(exist_ok=True)
cp = os.pathsep.join(str(deps / i["filename"]) for i in lock)
compile_run = subprocess.run(
    [
        str(args.java_bin / "javac"),
        "--release",
        "11",
        "-cp",
        cp,
        "-d",
        str(classes),
        str(here / "OrekitPcProbe.java"),
    ],
    capture_output=True,
    text=True,
    check=True,
)
observations = []
for radius in [cdm["hbr_m"] / 2, cdm["hbr_m"], cdm["hbr_m"] * 2]:
    values = [*rotated, *sigmas, radius]
    start = datetime.datetime.now(datetime.UTC)
    proc = subprocess.Popen(
        [
            str(args.java_bin / "java"),
            "-cp",
            str(classes) + os.pathsep + cp,
            "OrekitPcProbe",
            *[repr(float(x)) for x in values],
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    end = datetime.datetime.now(datetime.UTC)
    assert proc.returncode == 0, stderr
    observed = json.loads(stdout)
    local = disk_probability(mean, cov, radius, order=128)["pc"]
    observations.append(
        {
            "radius_m": radius,
            "input": [float(x) for x in values],
            "orekit": observed,
            "local_gl128_pc": local,
            "relative_difference": abs(observed["tighter_pc"] - local) / local,
            "pid": proc.pid,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "elapsed_s": (end - start).total_seconds(),
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    )
assert all(
    a["orekit"]["tighter_pc"] < b["orekit"]["tighter_pc"]
    for a, b in pairwise(observations)
)
receipt = {
    "scope": "projected_2d_pc_feasibility_probe",
    "orekit_version": "13.1",
    "java_version": subprocess.check_output(
        [str(args.java_bin / "java"), "-version"], stderr=subprocess.STDOUT, text=True
    ).strip(),
    "source_path": str(source),
    "argonavis_source_commit": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "preprocessor_sha256": hashlib.sha256(
        (root / "src/argonavis/services/epic49_cdm_reference.py").read_bytes()
    ).hexdigest(),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "java_source_sha256": hashlib.sha256(
        (here / "OrekitPcProbe.java").read_bytes()
    ).hexdigest(),
    "dependencies": lock,
    "geometry": {
        "mean_plane_m": mean.tolist(),
        "covariance_plane_m2": cov.tolist(),
        "eigenvalues_m2": eigen.tolist(),
        "proper_rotation": vectors.tolist(),
        "rotated_mean_m": rotated.tolist(),
        "hbr_m": cdm["hbr_m"],
    },
    "reported_pc": cdm["reported_pc"],
    "fresh_local_baseline_pc": result["baseline"]["pc"],
    "density_sensitivity_status": "incomplete_source_pc_mismatch",
    "observations": observations,
    "qualifications": {
        "real_orekit_invoked": True,
        "shared_geometry_preprocessor": True,
        "independent_full_cdm_frame_validation": False,
        "full_eop_or_trajectory_propagation": False,
        "argus_demo_runtime_integrated": False,
        "physical_accuracy_claim": False,
    },
    "note": "Nominal4.5m uses real released NASA input. Half/double-radius cases are diagnostic perturbations, not additional historical events. Agreement bounds numerical implementation under shared geometry, not operational accuracy.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
print(
    json.dumps(
        {
            "nominal_orekit_pc": observations[1]["orekit"]["tighter_pc"],
            "local_pc": observations[1]["local_gl128_pc"],
            "relative_difference": observations[1]["relative_difference"],
            "real_java_processes": len(observations),
        },
        indent=2,
    )
)
