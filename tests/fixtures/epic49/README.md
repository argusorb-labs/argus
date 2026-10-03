# T-388-A1 numerical packet

This is a research evidence packet, with no API, UI or production wiring. Actual Grok review is pending. `inputs.json` is the complete frozen input; `computed.json` and `independent.json` are derived outputs, never read by the evaluator. `receipt.json` binds the calculation commit, file hashes, commands and measured validation. Only `state_si` is authoritative for computation; `position_m` and `velocity_mps` duplicate its components for inspection.

From the source worktree, using the installed numerical environment:

```sh
PYTHONPATH=. /Users/yong/projects/substratum/argus/.venv/bin/python scripts/build_epic49_packet.py --output /private/tmp/epic49-recomputed.json --reference-output /private/tmp/epic49-rechecked.json
cmp tests/fixtures/epic49/computed.json /private/tmp/epic49-recomputed.json
cmp tests/fixtures/epic49/independent.json /private/tmp/epic49-rechecked.json
PYTHONPATH=. /Users/yong/projects/substratum/argus/.venv/bin/python -m pytest -q
/Users/yong/projects/substratum/argonavis/.venv/bin/ruff check services/demo_numerics.py scripts/build_epic49_packet.py scripts/epic49_reference.py tests/test_demo_numerics.py
/Users/yong/projects/substratum/argonavis/.venv/bin/ruff format --check services/demo_numerics.py scripts/build_epic49_packet.py scripts/epic49_reference.py tests/test_demo_numerics.py
```

The reference command independently integrates every frozen body and burned candidate using DOP853 and a separately written force law. It refines minima using bounded distance minimization at a finer bracket spacing, compares unburned RK4 10/5/2.5-second steps, and verifies polar Gauss–Legendre Pc against adaptive Cartesian disk quadrature. Same-state quadrature error is separated from trajectory error. It also recomputes magnitude, burn-time and initial-state perturbations and renamed/reordered options. Checks raise on failed tolerances; output hashes exclude timing metadata. Exact byte reproducibility is established on the recorded Python/NumPy/SciPy/SGP4 versions; other versions should use the stated numerical tolerances.

The original one-pass freeze can be reproduced into a separate file (do not replace reviewed inputs):

```sh
PYTHONPATH=. /Users/yong/projects/substratum/argus/.venv/bin/python scripts/build_epic49_packet.py --freeze-from-archive /Users/yong/projects/substratum/argus/data/spacetrack_raw_tle/tle2019.txt --inputs /private/tmp/epic49-refrozen.json
cmp tests/fixtures/epic49/inputs.json /private/tmp/epic49-refrozen.json
```

The archive is opened read-only. The stream hashes every byte and matches only adjacent line-1/line-2 catalog fields. There are 7,881,846 matching adjacent pairs and 3,219 nonmatching/unpaired line-2 records; the earlier broader 7,885,065 count must not be represented as matching adjacent pairs. Selection keeps the latest element per ID with epoch in the three days through 2019-09-02 10:00 UTC. This is a **retrospective epoch cutoff**, with publication availability unverified. The pinned historical pair is deliberately preserved rather than silently replaced by the latest elements; the later 44278 element is a separate sensitivity input.

The real catalog uses mean-motion radial overlap with 302–345 km, perigee at least 100 km, apogee at most 2,000 km, then ranks by refined SGP4 minimum to the pinned Aeolus trajectory over 10:02–12:02 UTC. Mean motion converts from rad/min to rad/s before computing the Kepler radial envelope. 16,991 age-eligible latest IDs reduce to 99 usable geometrically relevant IDs after excluding the two event objects, 16,889 radial non-overlaps and one propagation exclusion. The nearest 32 are frozen, with all lines, IDs, raw hashes and a full subset hash; 67 are truncated. This deterministic selection is bounded and retrospective, not a global catalog or a completeness claim.

Historical geometry uses split-Julian-date WGS72 SGP4 and continuous refinement: 2,592.260912 m at **2019-09-02 11:02:41.662 UTC**, relative speed 14,403.873351 m/s. The separate later 44278 element gives 2,880.302343 m at 11:02:41.614 UTC. The dynamical experiment starts from SGP4 states at 10:02 and propagates all bodies with the same frozen-axis two-body/J2 model. Its no-burn range is 2,745.069695 m; it is a separate model experiment, not a replacement for SGP4 geometry. None of its 32 catalog trajectories has a refined minimum within the 50 km volume for any tested option. Pc is null and candidate recommendations abstain because historical covariance is absent. Real-object radii are explicitly assumed, unverified and unused for historical Pc.

The **constructed operational benchmark** has its own catalog: one explicitly synthetic crossing object, with no real NORAD identity. No historical catalog is mixed into it. Common epoch is 2019-09-02 10:02 UTC; all full SI state vectors are frozen. The primary and secondary seeds at t=3,600 s are circular-like perpendicular LEO crossing states at radius 7,000 km with a declared 100 m radial offset. They are integrated backwards using DOP853. The synthetic object's seed is the actual positive along-track candidate position at t=5,400 s, with a declared perpendicular circular-like velocity, then integrated backwards to the same epoch. These are declared fixture inputs, not discovered debris or historical risk. No input is changed after freezing to match outputs.

The measured constructed outcomes are:

| Option / impulse at common epoch | Primary minimum (m) | Primary assumed Pc | Synthetic minimum (m) | Synthetic assumed Pc | Draft evaluation |
| --- | ---: | ---: | ---: | ---: | --- |
| No burn | 100.000003 | 7.878073e-4 | 3,799.381029 | 7.754725e-67 | rejected |
| Along +0.30 m/s | 2,963.202895 | 9.847800e-47 | 0.000002591 | 3.951980e-4 | rejected by synthetic crossing |
| Along −0.30 m/s | 3,027.194360 | 6.568193e-51 | 7,597.758555 | 6.823129e-255 | eligible within assumed bounded scope |
| Cross +0.30 m/s | 167.728836 | 6.572372e-4 | 3,716.026185 | 4.085001e-64 | rejected |
| Cross −0.30 m/s | 167.638490 | 6.574290e-4 | 3,882.585337 | 1.295859e-69 | rejected |

Covariances are **assumed independent position covariances at each TCA**, in each body's own RIC: sigmas 100/200/100 m. They are rotated independently into common axes and then the relative-velocity encounter plane. They are not estimated, empirically calibrated, or transported from the burn epoch. Radius sums are 4+4=8 m for the primary conjunction and 4+1=5 m for the synthetic object. Scenario alert Pc >1e-5 only alerts; rejection Pc >1e-4 rejects. The simulated delta-v limit is 0.5 m/s. These are declared demonstration constraints, not operator policy. Every result is a draft pending human review, never a spacecraft command.

Finite/SPD checks reject invalid covariance. Low relative speed (≤1,000 m/s), boundary minima, non-brief encounters, or longitudinal miss inconsistent with TCA abstain. Covariance missing outside assumed mode and events outside the 50 km screened volume return null Pc with explicit status. Tiny evaluated probabilities retain finite `log_pc`; machine underflow is labeled and never used to hide missing evaluation. Encounter approximation requires duration `(HBR + 3 sigma_max)/speed` ≤1% of local dynamical time and longitudinal miss ≤1 m; this is a bounded applicability check, not full flight qualification. Thirty-second grids find brackets, never serve as distance gates; 60/30/10-second bracket convergence and finer independent reference minimization are tested.

Model limitations: frozen TEME axes/J2 symmetry axis, no drag, Earth-axis precession, higher gravity harmonics, maneuvers of other real objects, covariance dynamics or OD uncertainty validation. RK4→DOP853 agreement measures numerical consistency within this model; it does not establish real-orbit prediction accuracy. The horizon is two hours, with a finite dated catalog; there is no global safety claim. No measured operational savings, calibrated historical risk, investor story, operator messages or commands are provided.
