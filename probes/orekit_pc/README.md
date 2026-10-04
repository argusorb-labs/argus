# Real Orekit projected-2D collision-probability probe

Status: executed feasibility probe; not integrated into the demo runtime. Owner: Codex, T-388-TOOLS-OREKIT.

This probe uses the real Orekit13.1 `Patera2005` class in physical Java processes. NASA's released ICON44628 / PSLV DEB27127 CDM is parsed by the existing source-pinned Argonavis kernel. Its per-object velocity/covariance preparation and encounter-plane projection are reused. A proper eigenbasis rotation diagonalizes the2D covariance; projected position, positive eigen-sigmas and combined radius4.5m are passed to Orekit's scalar API. No result is copied from a historical receipt.

Actual nominal output:

| Calculation | Pc |
| :--- | :--- |
| NASA printed historical value | 0.0001601 |
| Fresh local Gauss-Legendre128 | 0.00016012324260402853 |
| Orekit Patera2005, tighter numerical integration | 0.0001601232426039784 |

Measured relative difference:3.129916969460719e-13. This is numerical agreement for the shared projected inputs, not measured physical/operational accuracy or an independent full-CDM frame check. Default-versus-tighter integration and half/double-radius diagnostic cases are preserved in [result.json](result.json), including actual PIDs, timestamps, stdout and exit codes. Radius perturbations are constructed diagnostics, not extra historical events. NASA density sensitivity mismatch remains unresolved and cannot clear action.

## Reproduce

Requirements: Python with NumPy and the existing Argonavis scientific dependencies; a Java JDK; source checkout at Argonavis1e50f863. No changes to application dependency configuration are required.

```bash
python probes/orekit_pc/fetch_dependencies.py --deps /private/tmp/argus-orekit-deps
python probes/orekit_pc/run_probe.py \
  --argonavis-root /path/to/argonavis-at-1e50f863 \
  --java-bin /path/to/jdk/bin \
  --deps /private/tmp/argus-orekit-deps \
  --output /private/tmp/orekit-probe-result.json
```

[dependencies.json](dependencies.json) pins Orekit13.1 and Hipparchus4.0.1 artifacts, public Maven URLs and SHA-256 fingerprints. Jars/class output remain outside Git. The observed runtime was Homebrew OpenJDK26.0.2.1; compilation targets Java11. Code validates finite inputs, positive sigmas/radius, output probability range and radius monotonicity. Java process timeout cleanup is bounded. Actual reproducibility and method-pair qualification need further cases before any general equivalence claim.

## Evidence and limits

- [Official Patera API reference](https://www.orekit.org/static/apidocs/org/orekit/ssa/collision/shorttermencounter/probability/twod/Patera2005.html) describes the scalar rotated-encounter inputs. Its current rendered version may differ from the pinned13.1 jar; actual13.1 signatures were checked with `javap`.
- [Official short-term method assumptions](https://www.orekit.org/site-orekit-13.1/apidocs/org/orekit/ssa/collision/shorttermencounter/probability/twod/AbstractShortTermEncounter2DPOCMethod.html): short linear encounter, Gaussian position uncertainty, spherical bodies and deterministic velocities.
- [Pinned Maven POM](https://repo.maven.apache.org/maven2/org/orekit/orekit/13.1/orekit-13.1.pom) identifies the public artifact and dependencies.

Shared geometry preprocessing means this is an independently executed probability-method comparison, not independent geometry/frame validation. No EOP dataset was loaded; no propagation, maneuver optimization, catalog screening, vendor service, flight command or live model planner ran. The existing app runtime still reports zero live model calls. Next integration must bind this verified subset of Orekit capability to task-specific qualification and actual adapter readiness rather than making every Orekit capability executable.
