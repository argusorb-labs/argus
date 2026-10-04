# Released NASA ICON / PSLV DEB CDM evidence

Status: author packet ready for physical independent review. Baseline reproduced;
optional corrected **published Pc comparison incomplete**. No operational replay,
propagation, maneuver recommendation, covariance calibration or flight authority.

## Source and extraction

[NASA-SP-20205011318/REV2-VOL2, August 2026](https://www.nasa.gov/wp-content/uploads/2026/08/ca-handbook-volume2-nasa-sp-20205011318-rev2-vol2.pdf),
Appendix N. PDF pages 95–97 (printed 87–89) contain the CDM; PDF94 footnote6
records public clearance April 03, 2024, Orbital Data Request #24-001-1.
PDF92–94 give N-11 through N-22; PDF98–99 give comparison invariants.
Only released numerical data and attribution are checked in, not the full book.

`source_cdm.kvn` preserves printed tokens, units, comments and the folded
MESSAGE_ID/DCP lines. pypdf 6.19.0 text extraction removed only page furniture;
`provenance.json` gives PDF SHA256, page text hashes, KVN line mapping and
local PDFKit render proof hashes. Author inspected all six rendered pages and
compared states, all 42 covariance entries and DCP tokens. Those temporary
renders are evidence artifacts, not product dependencies or checked-in prose.
The raw PDF hash is cc5acf7de44526d7bfc187f21e457e5969bedffdacaef69227ba619637a3b6dd.

## Frame interpretation and derivation

[CCSDS 500.0-G-4](https://ccsds.org/Pubs/500x0g4.pdf), §5.2.2.3 defines state
velocity as coordinate derivatives in the specified coordinate system.
[CCSDS 508.0-P-1.1](https://ccsds.org/wp-content/uploads/gravity_forms/9-6f599803174a64f5da08b9814720b5c4/2025/02/508x0p11.pdf)
(March 2024 draft, corroboration rather than the version governing this 2022
message), table3-4 and §5.2.2 specify REF_FRAME and TCA states, and annexB5
explicitly calls for the omega-cross-position term between rotating/inertial
frames. NASA N-14/15 specifies RTN axes from inertial r and v.

[IERS terrestrial-to-celestial derivation](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote32/tn32.pdf?__blob=publicationFile),
chapter5, has r_I = Q(t) R(t) W(t) r_T. Differentiating gives
v_I = U v_T + U_dot r_T. For nominal z-axis spin with fixed pole and no
polar-motion/LOD rates, U_dot r_T = U (omega × r_T), so physical velocity
expressed in the instantaneous terrestrial orientation is **v_T + omega × r_T**.
The adopted nominal omega is 7.29211514670698e-5 rad/s (ERA sidereal rate).
This is a documented approximation, not a full EOP/time-dependent ITRF-to-J2000
transform. No TEME engine is used.

At one instant choose U(TCA) = identity: inertial vectors can be expressed in
axes aligned with ITRF **at that instant**. This does not make those axes inertial
through time. Any constant proper rotation of all states, covariances and DCP
vectors leaves determinants, eigenvalues and the disk probability unchanged.
Hence no J2000 orientation is needed for these invariant comparisons. Missing
EOP/pole-rate corrections limit the result to the shown source-rounding agreement;
it is not exact frame transformation or observation accuracy.

Each object's own r and physical v defines its own RTN rotation. Rotate each
6x6 covariance with blockdiag(M,M), NASA N-12/13, then sum the position blocks
for N-17. The position marginal contains all RTN off-diagonal terms; the velocity
and position/velocity blocks are preserved but not used by the 2D calculation.
The encounter plane is normal to relative **physical** velocity. The computed
RIC values match the CDM's independently rounded entries within 0.051 m or m/s.
Full states imply a 3.350138 m longitudinal residual and a -0.0002512203 s
linear closest-approach offset. The 2D plane integrates over the linear encounter;
it does not replace the full positions with the rounded printed relative fields.
The receipt retains this residual and does not pretend the epochs were shifted
or that an operational ephemeris was supplied.

## Risk and the source discrepancy

Production reuses only the unchanged accepted `disk_probability` polar
Gauss-Legendre integral (order96), after this module validates inputs. Independent
reference constructs RTN and encounter axes differently and uses nested adaptive
Cartesian integration. Both share the source interpretation and short Gaussian
encounter assumptions; agreement is numerical verification, not calibration.

Provider CDM Pc: **1.601e-4**, FOSTER-1992. Computed baseline:
**1.6012324260402769e-4**, agreeing within the printed rounding interval.
N-25 eigenvalues and determinant also agree at rounding precision.

NASA N-18 uses the two supplied density sigmas and independently rotated DCP
position vectors: C_relative = C_p + C_s - sigma_p sigma_s
(G_p G_s^T + G_s G_p^T). This sourced shared-global-density hypothesis gives
**8.040404963339254e-5**, adaptive **8.04040496333715e-5**. Corrected determinant
8.101968747630188e13 m6 and sigmas [13.9799326,189.587606,3396.09837] m agree
with N-28/PDF99 reference invariants within stated rounding tolerances.
No calibrated correlation coefficient is invented or fitted.

PDF94 visibly prints **8.04e-4** while saying the probability is reduced by
about two; those printed probabilities instead imply a ratio of 5.025.
The reconstruction ratio is 0.50213853. A source exponent typo is a plausible
**inference**, not a confirmed erratum. The packet preserves the 8.04e-4 target,
reports `incomplete_source_pc_mismatch`, and does not claim to reproduce it.
Covariance rounding and the tiny longitudinal offset do not explain a factor10.
Physical Grok review must adjudicate this optional comparison before product use.

## Interface and rebuild

- `parse_cdm(text) -> dict`: schema_version1; message_id, creation_date, tca,
  reported_pc/method, combined hbr_m, two objects. Objects contain ITRF state in
  m/m/s, RTN symmetric6x6 covariance in m2/m2/s/m2/s2, density_sigma,
  DCP position/velocity sensitivities and original string fields/comments.
- `evaluate_cdm(parsed, orientation=None) -> dict`: reported value, separate
  baseline and correlation_sensitivity, physical states, own RTN bases, common
  covariances, encounter plane, eigenvalues/determinants, geometry and explicit
  model_conditions. Positive definite full covariance required; no PSD repair.
  Missing DCP abstains from correction; invalid supplied DCP rejects the input.
  Unknown frames/units, missing/nonfinite/indefinite covariance, invalid HBR,
  nonfinite states, undefined orbit axes and near-zero relative velocity reject.
- `scripts.epic49_cdm_reference.evaluate(parsed) -> dict`: independent reference
  for this valid nonzero-miss fixture, including adaptive integration estimates.
- `python -m scripts.build_epic49_cdm_packet`: deterministic JSON rebuild from
  the compact source KVN/provenance, no network. Optional `--output PATH` and
  `--verify-pdf PATH` (hash checks the locally supplied original PDF).

Outputs: `source_cdm.json`, `computed.json`, `independent.json`, `receipt.json`.
Receipt pins source and algorithm hashes, errors/tolerances/versions, source
comparisons and all original base regular files plus accepted science-anchor
blobs. UI/API does not consume these files yet; later SPEC must adapt the
reviewed interface. EGM96/JBH09/etc remain source input-model metadata only.
No pre-TCA operational ephemeris, burns, mission constraints, catalog or named
history was manufactured; no anonymous Kelvins join is made.

## Execution record

Physical Codex T-388-R2-NUM, supplied isolated source worktree/base2db2eaa.
No subagents or simulated reviews. Parent owns physical Grok review/acceptance.
RED: 20 tests failed on absent CDM module; independent-reference RED failed on
absent reference; builder RED failed on absent builder with26 other tests green.
Raw initial RED receipts are hashed in `verification.json`.

Ruling: use supplied plan/worktrees/physical handoff instead of builtin role
helpers — the user explicitly supplied execution and parent-owned review.
Ruling: symmetrize analytically symmetric rotated products at their creation —
5.82e-11 m2 asymmetry is floating multiplication roundoff, not source PSD repair;
raw covariance validation remains strict and no eigenvalue clipping is used.
Ruling: replace the inapplicable borrowed round1 longitudinal cutoff with the
linear encounter offset plus duration/dynamical-time condition — the source
requires a full encounter-plane integral; record the actual residual explicitly.
Ruling: corrected published-Pc reproduction is optional/incomplete — N-18 and
two integrators agree with corrected invariants, not the printed exponent. Do
not change data/HBR/thresholds to fit it. The baseline reproduction is complete.
