# Argus Internal Knowledge: Space Operations Tools

Knowledge role: **internal tool-selection prior**. Language: English. Evidence checked: 2026-10-04. Maintainer: Codex; reviewer: Yong. Capability details remain subject to review and integration verification.

Start with [the capability index](INDEX.md). The [structured catalog](catalog.json) contains the same 20 tool/service records for future retrieval. This is a representative inventory, not a market-share ranking or an exhaustive list of ground networks, space-weather, thermal, power, payload scheduling or deep-space tools.

## Purpose and use

Use this knowledge to identify candidate capabilities, prerequisites and workflow changes for a task. Combine it with the task requirements and current adapter, account, data and health checks before execution. A shared capability tag does not establish equivalence. Orekit and Kayhan are not interchangeable as whole products.

The catalog is a knowledge artifact, not an executable registry or approved API schema. It has not been connected to Argus runtime retrieval. No software installation, provider request, credential access, ephemeris upload, reservation, uplink or model integration was performed. Existing demo use of sgp4 does not verify new adapters for this inventory.

## Evidence interpretation

- Each card preserves documented capabilities, prerequisites, outputs, interfaces, restrictions, access conditions, workflow implications and primary sources. Input prerequisites and workflow implications are analysis based on documentation and physical requirements, not verified API parameter schemas.
- All records are verified at the **public-documentation** level only. Exact endpoint schemas, Argus calls, account permissions and runtime health remain unverified. Runtime availability is unknown. Product descriptions are not independent accuracy tests or SLAs.
- References differ in age: Orekit 13.0 overview and 13.1 Pc API are not asserted to be the latest; the NASA GMAT page labels R2026; the referenced FreeFlyer Runtime documentation specifies Mission tier. LeoLabs material is older; the EU SST portfolio is its 2024 third edition, with older REST material. Retrieval date is not publication date.
- The Kayhan Dynamics URL has an Elements page title and Dynamics body; screening SDK documentation still uses Pathfinder. Confirm the current product, schema and entitlement without guessing a rebranding or API mapping.
- A historical NASA 2022 CDM cannot be combined with a current catalog and presented as historical full-catalog screening unless the service verifies matching historical coverage. A numerical propagator cannot create missing observations, covariance or operator ephemerides.
- Cost, latency, accuracy and availability have not been measured. Do not rank execution choices using unverified marketing metrics. Pin and verify exact software licenses during deployment.

## Tool cards

<a id="celestrak"></a>

### CelesTrak (`celestrak`)

- **Capabilities:** catalog_gp
- **Required inputs:** GP/OMM/TLE query criteria, objects and time range
- **Outputs:** GP elements and catalog-related data
- **Documented interface:** Public HTTP downloads and queries, subject to usage policy
- **Access prerequisites:** public_query_policy
- **Limitations:** Format compatibility does not improve orbital accuracy. GP elements must use the matching SGP4 model; operational covariance is not guaranteed.
- **Workflow implications:** Retrieve data, check epoch and element theory, then propagate with SGP4. These data alone do not establish high-accuracy maneuver screening.
- **Sources:** [Primary source 1](https://celestrak.org/NORAD/documentation/gp-data-formats.php); [Primary source 2](https://www.celestrak.org/usage-policy.php).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="space_track"></a>

### Space-Track (`space_track`)

- **Capabilities:** catalog_gp, cdm_delivery
- **Required inputs:** Account, query permissions, objects and time criteria
- **Outputs:** GP, historical data and related CDM products within authorized access
- **Documented interface:** Authenticated HTTP API; CDM products require appropriate permissions
- **Access prerequisites:** account_and_product_permissions
- **Limitations:** Public API documentation does not establish that the current account can access a particular spacecraft CDM. Verify rate limits and use restrictions.
- **Workflow implications:** Check product permissions and data coverage before requesting. Treat catalog elements and CDMs as distinct evidence types.
- **Sources:** [Primary source 1](https://www.space-track.org/documentation); [Primary source 2](https://www.space-track.org/documents/Spacetrack_Handbook_for_Operators.pdf).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="discos"></a>

### ESA DISCOSweb (`discos`)

- **Capabilities:** object_metadata
- **Required inputs:** Object identifiers and account/API access
- **Outputs:** Object registration, launch and physical-property metadata
- **Documented interface:** Portal and API; exact fields and token requirements need integration verification
- **Access prerequisites:** api_access_to_verify
- **Limitations:** Object metadata does not automatically establish an attitude-dependent hard-body radius (HBR) or current covariance.
- **Workflow implications:** Enrich object information with field-level provenance and missing-data flags; do not fabricate absent parameters.
- **Sources:** [Primary source 1](https://discosweb.esoc.esa.int/).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="sgp4"></a>

### python-sgp4 / SGP4 (`sgp4`)

- **Capabilities:** gp_propagation
- **Required inputs:** Compatible TLE/OMM GP elements and target epoch
- **Outputs:** TEME position and velocity, with propagation error codes
- **Documented interface:** Python library; pin the underlying implementation and version
- **Access prerequisites:** open_source_license_to_pin
- **Limitations:** Not a universal substitute for high-fidelity propagation. GP data do not supply operational covariance; frame conversion must be explicit.
- **Workflow implications:** Propagate with the matching GP theory, check epoch and errors, and convert frames when required. Missing covariance blocks quantitative Pc estimation.
- **Sources:** [Primary source 1](https://github.com/brandon-rhodes/python-sgp4); [Primary source 2](https://www.celestrak.org/software/tutorials/sgp4.php).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="orekit"></a>

### Orekit (`orekit`)

- **Capabilities:** state_propagation, orbit_determination, frames_time, cdm_pc, event_geometry
- **Required inputs:** State, observations or CDM; frames and time scales; required EOP, gravity and celestial data; covariance and HBR for Pc
- **Outputs:** Orbits, estimation results, transformations, geometric events and Pc under applicable assumptions
- **Documented interface:** Java API; actual bridge and deployment require verification
- **Access prerequisites:** open_source_license_to_pin
- **Limitations:** Short-term two-dimensional Pc assumes conditions including linear relative motion and independent Gaussian position uncertainty. Record environmental data and calculation method.
- **Workflow implications:** Load environmental data, validate prerequisites, select propagation/estimation/Pc methods, and return versioned outputs with assumptions.
- **Sources:** [Primary source 1](https://www.orekit.org/site-orekit-13.0/index.html); [Primary source 2](https://www.orekit.org/site-orekit-13.1/apidocs/org/orekit/ssa/collision/shorttermencounter/probability/twod/AbstractShortTermEncounter2DPOCMethod.html).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="tudat"></a>

### Tudat / TudatPy (`tudat`)

- **Capabilities:** state_propagation, orbit_determination, variational_equations
- **Required inputs:** Initial state, environment and force models, integration and termination settings; observations and estimated parameters for estimation
- **Outputs:** State histories, variational-equation results, and parameter/state estimates
- **Documented interface:** Python interfaces with a C++ core
- **Access prerequisites:** open_source_license_to_pin
- **Limitations:** Propagation and estimation support does not establish catalog screening or mission-constraint coverage.
- **Workflow implications:** Build the environment and dynamics, propagate or estimate, and retain residuals and configuration. Pc needs a separate capability.
- **Sources:** [Primary source 1](https://py.api.tudat.space/en/latest/); [Primary source 2](https://docs.tudat.space/en/latest/user-guide/state-propagation/propagation-setup.html).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="gmat"></a>

### NASA GMAT (`gmat`)

- **Capabilities:** state_propagation, trajectory_design, trajectory_optimization, navigation
- **Required inputs:** Mission script, initial orbit, environment models, objectives and constraints
- **Outputs:** Trajectories, optimization/targeting and analysis results
- **Documented interface:** GUI, mission scripts, Python and Java; verify modules in the selected version
- **Access prerequisites:** open_source
- **Limitations:** Mission design and navigation software is not itself an external conjunction catalog service. Verify API features by version.
- **Workflow implications:** Generate mission configuration, run it, check convergence and constraints, and export trajectories for independent screening.
- **Sources:** [Primary source 1](https://software.nasa.gov/software/GSC-19640-1); [Primary source 2](https://data.nasa.gov/dataset/general-mission-analysis-tool-project).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="spice"></a>

### NAIF SPICE (`spice`)

- **Capabilities:** frames_time, ephemeris_geometry, event_geometry
- **Required inputs:** Kernels such as SPK/CK/PCK/FK/LSK covering the target and epoch, plus object and frame identifiers
- **Outputs:** State, attitude, time/frame transformations and geometric events
- **Documented interface:** Toolkits for C, Fortran, IDL and MATLAB; third-party Python wrappers need separate verification
- **Access prerequisites:** toolkit_terms_to_verify
- **Limitations:** Computes geometry from available kernels; does not automatically determine arbitrary spacecraft orbits or generate operational trajectories.
- **Workflow implications:** Check kernel types, versions and coverage, compute geometry, and retain the kernel manifest. Missing coverage blocks the calculation.
- **Sources:** [Primary source 1](https://naif.jpl.nasa.gov/naif/toolkit.html); [Primary source 2](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/info/mostused.html).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="stk"></a>

### Ansys STK / Astrogator (`stk`)

- **Capabilities:** event_geometry, access_analysis, trajectory_design
- **Required inputs:** Scenario, orbits, sensors and sites; engine models and constraints for maneuver design
- **Outputs:** Visibility/contact windows, scenario analysis, and Astrogator trajectory/maneuver results
- **Documented interface:** STK Python API and Engine; Engine documentation supports Windows/Linux; module licenses required
- **Access prerequisites:** licensed_modules
- **Limitations:** Verify installation, product modules and Engine entitlement. Geometric access does not mean a physical station is reserved.
- **Workflow implications:** Build the scenario, run analysis/targeting, and export results. Station reservation is a separate service step.
- **Sources:** [Primary source 1](https://help.agi.com/stkdevkit/Content/python/pythonGettingStarted.htm); [Primary source 2](https://www.ansys.com/content/dam/amp/2022/june/webpage-requests/stk-product-page/brochures/stk-premium-space-brochure.pdf).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="odtk"></a>

### Ansys ODTK (`odtk`)

- **Capabilities:** orbit_determination, covariance_analysis
- **Required inputs:** Tracking observations, measurement models, initial state and uncertainty
- **Outputs:** Filtering/estimation and orbital-uncertainty products
- **Documented interface:** Application automation; the referenced COM integration is Windows-specific; verify other deployments separately
- **Access prerequisites:** licensed_runtime_to_verify
- **Limitations:** Requires observations and measurement-quality configuration. Do not transfer STK Python/platform support claims to ODTK.
- **Workflow implications:** Configure observations and priors, filter and check quality, then export orbit/covariance products for risk analysis.
- **Sources:** [Primary source 1](https://help.agi.com/odtk/Content/od/odtkIntegratingTop.htm); [Primary source 2](https://help.agi.com/odtk/content/od/ODObjectsSatelliteOrbitUncertainty.htm).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="freeflyer"></a>

### FreeFlyer (`freeflyer`)

- **Capabilities:** state_propagation, orbit_determination, mission_analysis
- **Required inputs:** Mission Plan/script, orbit or observations, and model settings
- **Outputs:** Propagation, estimation and mission-analysis results
- **Documented interface:** Referenced Runtime API documents C/C++, C#, Java and Python; restricted to Mission tier in that reference
- **Access prerequisites:** mission_tier_per_reference
- **Limitations:** Recheck documentation version, tier and current license. Local deployment and generic interchangeability have not been verified.
- **Workflow implications:** Generate a Mission Plan, invoke the licensed runtime, check exceptions and estimation/analysis quality, then extract results.
- **Sources:** [Primary source 1](https://www.ai-solutions.com/_help_Files/using_the_runtime_api.htm); [Primary source 2](https://www.ai-solutions.com/_help_Files/spacecraft_od_setup.htm).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="kayhan_dynamics"></a>

### Kayhan Dynamics (`kayhan_dynamics`)

- **Capabilities:** state_propagation, orbit_determination
- **Required inputs:** State or observations and model configuration; exact interface schema to obtain
- **Outputs:** Orbit simulation and estimation results; exact output contract to verify
- **Documented interface:** Official product page lists Managed API, SDK and local deployment
- **Access prerequisites:** commercial_entitlement_to_verify
- **Limitations:** Product descriptions do not establish endpoints, entitlement, accuracy or covariance fields. The page title says Elements while the body says Dynamics; verify naming.
- **Workflow implications:** Verify current interface, deployment and product access; configure computation; validate outputs and model/data versions.
- **Sources:** [Primary source 1](https://kayhan.space/products/dynamics); [Primary source 2](https://kayhan.space/products).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="kayhan_screening"></a>

### Kayhan SDK screening service (documented as Pathfinder) (`kayhan_screening`)

- **Capabilities:** ephemeris_screening
- **Required inputs:** Account permissions, primary NORAD ID and ephemeris in a supported format
- **Outputs:** Screening job identifier, completion status and conjunction results
- **Documented interface:** Public SDK/CLI documents upload, submission/waiting and queries; authentication configuration includes password and M2M modes
- **Access prerequisites:** authenticated_product_access_to_verify
- **Limitations:** Verify older product naming and interfaces. Historical catalog coverage and maneuver-planning API access are unconfirmed. Represent maneuver discontinuities in separate blocks.
- **Workflow implications:** Validate/upload ephemeris, submit, await/query and retrieve events. Changing screening providers requires checking the same time-window coverage.
- **Sources:** [Primary source 1](https://kayhan-oss.gitlab.io/kayhan-sdk/pages/CLI/gettingstarted.html); [Primary source 2](https://kayhan-oss.gitlab.io/kayhan-sdk/pages/configuration.html).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="leolabs"></a>

### LeoLabs Collision Avoidance (`leolabs`)

- **Capabilities:** cdm_delivery, ephemeris_screening, secondary_tracking
- **Required inputs:** Operator ephemeris/object and authorization; actual job interface to verify
- **Outputs:** CDMs/events, on-demand screening, secondary tracking and orbit/covariance-related products
- **Documented interface:** Official product material describes cloud services and a REST API
- **Access prerequisites:** commercial_access_to_verify
- **Limitations:** Referenced product material is under a 2021 URL. Verify current schema/SLA, catalog population/coverage and historical support. Do not treat old timing claims as guarantees.
- **Workflow implications:** Submit compatible ephemeris, retrieve input-bound screening results and retain catalog/observation epochs. Additional tracking is a separate request with external effects.
- **Sources:** [Primary source 1](https://www.leolabs.space/wp-content/uploads/2021/06/LeoLabs-Collision-Avoidance.pdf).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="eu_sst"></a>

### EU SST services (`eu_sst`)

- **Capabilities:** cdm_delivery, collision_assessment, reentry_assessment, fragmentation_analysis
- **Required inputs:** Registration approval and service-specific operator configuration/orbit inputs
- **Outputs:** CDMs, risk reports, re-entry and fragmentation-related products
- **Documented interface:** Portal; older official material describes REST access; verify current API contracts
- **Access prerequisites:** registration_and_service_agreement
- **Limitations:** Free service still requires approval and configuration. Eligibility differs by service. Full-catalog bulk access is not established.
- **Workflow implications:** Confirm registration and service configuration, submit authorized trajectories or retrieve products, and involve the operations center for high-interest events.
- **Sources:** [Primary source 1](https://www.eusst.eu/sites/default/files/documents/EUSST_Service_Portfolio.pdf); [Primary source 2](https://www.eusst.eu/wp-content/uploads/2020/11/EUSST_2WBR_16_11_2020.pdf).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="basilisk"></a>

### Basilisk (`basilisk`)

- **Capabilities:** spacecraft_dynamics, attitude_control_simulation, maneuver_simulation
- **Required inputs:** Spacecraft, actuators, sensors, initial orbit/attitude, control algorithms and environment
- **Outputs:** Coupled dynamics and attitude/actuator/control simulation histories
- **Documented interface:** Python configuration and C/C++ modules
- **Access prerequisites:** open_source_license_to_pin
- **Limitations:** Not a ready-made catalog screener or flight-command authorization system. Fidelity depends on the selected models and parameters.
- **Workflow implications:** Insert a candidate maneuver into the spacecraft model, simulate actuator/attitude behavior, check feasibility and return the trajectory for renewed screening.
- **Sources:** [Primary source 1](https://hanspeterschaub.info/basilisk/); [Primary source 2](https://www.hanspeterschaub.info/research-Attitude.html).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="yamcs"></a>

### Yamcs (`yamcs`)

- **Capabilities:** telemetry_access, alarms, commanding
- **Required inputs:** Deployed instance, mission database, telemetry links and user/command permissions
- **Outputs:** Parameters/packets, alarms, events and command-status data
- **Documented interface:** HTTP JSON/Protobuf, WebSocket and a Python client
- **Access prerequisites:** deployment_and_mission_permissions
- **Limitations:** Requires mission-specific database and links. Authorize read-only telemetry and command effects separately.
- **Workflow implications:** Read telemetry/alarms for analysis. Commands require explicit authority and mission review; successful computation does not automatically authorize uplink.
- **Sources:** [Primary source 1](https://docs.yamcs.org/yamcs-http-api/); [Primary source 2](https://docs.yamcs.org/yamcs-http-api/overview/).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="openc3"></a>

### OpenC3 COSMOS (`openc3`)

- **Capabilities:** telemetry_access, alarms, commanding, procedure_execution
- **Required inputs:** Target/packet definitions, plugins, interfaces and instance authentication
- **Outputs:** Telemetry, limits/logs, commands and procedure-execution data
- **Documented interface:** HTTP JSON-RPC, scripts and WebSocket streaming; some interfaces are Enterprise-only
- **Access prerequisites:** deployment_and_edition_permissions
- **Limitations:** Mission configuration is not generic plug-and-play. Documented methods that bypass checks do not grant Argus permission to use them.
- **Workflow implications:** Read mission-matched telemetry, analyze and draft procedures; execute procedures/commands only after separate permission checks.
- **Sources:** [Primary source 1](https://docs.openc3.com/docs/development/json-api); [Primary source 2](https://docs.openc3.com/docs/configuration/interfaces); [Primary source 3](https://docs.openc3.com/docs/development/streaming-api).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="aws_ground_station"></a>

### AWS Ground Station (`aws_ground_station`)

- **Capabilities:** contact_opportunities, contact_reservation
- **Required inputs:** Configured/authorized satellite, mission profile, IAM, time window and covering ephemeris
- **Outputs:** Contact opportunities, contact identifiers/status and associated data-transfer resources
- **Documented interface:** AWS API/SDK/CLI; documented TLE/OEM/azimuth-elevation workflows
- **Access prerequisites:** aws_account_onboarding_iam_and_billing
- **Limitations:** An available window does not guarantee reservation. Reservations affect resources and billing; check that the contact reaches SCHEDULED.
- **Workflow implications:** Validate ephemeris availability, list/select contacts, obtain reservation authority and confirm status or resolve conflicts. Recheck windows after trajectory changes.
- **Sources:** [Primary source 1](https://docs.aws.amazon.com/ground-station/latest/ug/contacts.html); [Primary source 2](https://docs.aws.amazon.com/ground-station/latest/ug/reserving-contacts-with-custom-ephemeris.html).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

<a id="drama_master"></a>

### ESA DRAMA / MASTER (`drama_master`)

- **Capabilities:** debris_flux, disposal_analysis, reentry_risk, mitigation_analysis
- **Required inputs:** Mission/spacecraft and orbit models, environmental data, software license and applicable version
- **Outputs:** Debris flux, mitigation/disposal and re-entry analyses
- **Documented interface:** ESA software portal; DRAMA depends on MASTER; programmatic interface to verify
- **Access prerequisites:** portal_license_application
- **Limitations:** Not a direct replacement for catalog screening of a current individual conjunction. Verify current license eligibility, models and standards versions.
- **Workflow implications:** Load environment/mission inputs, run applicable modules and interpret results under their models/standards. Regulatory compliance requires separate verification.
- **Sources:** [Primary source 1](https://www.sdo.esoc.esa.int/); [Primary source 2](https://conference.sdo.esoc.esa.int/proceedings/sdc9/paper/290).
- **Verification:** public documentation only; Argus calls unverified; account/runtime availability unknown.

## Planning implications for Argus

These are knowledge-use guidelines; detailed runtime contracts and implementation remain follow-on work.

1. Establish the task objective, time range, evidence, required outputs, accuracy/coverage requirements, deadline, cost and permission limits.
2. Retrieve candidate capabilities, then check current adapters, accounts, health and data. A documentation-only record does not permit execution.
3. The model may propose tool combinations and explain choices. The execution layer must validate contracts, task constraints and permissions. Missing inputs require retrieval or an evidence request.
4. A tool change requires checking affected dependencies: file/frame conversion, asynchronous waiting, cross-validation and post-maneuver screening. Identify invalidated downstream results and review drafts.
5. If no eligible implementation exists, return partial results and explicit gaps. Relaxing accuracy, scope or permissions changes the task requirements; it is not successful failover.

**Example A:** GP-only coarse position prediction suggests SGP4. Operational risk assessment without covariance requires more evidence; selecting Orekit alone does not establish higher accuracy.

**Example B:** A candidate trajectory followed by a screening service requiring OEM adds OEM generation/validation and maneuver-discontinuity blocks. A timeout may justify evaluating another screener, but its catalog and time coverage must be checked and the new request/result provenance retained.

**Example C:** A maneuver changes ground-contact windows. Recompute geometry and check actual station resources. An unavailable station may require a compatible site and revised steps. Contact reservations and flight commands have separate external effects and authorization requirements.

**Normal/failure/late-result trace:** select an eligible, currently available implementation, execute and validate; on failure, retrieve compatible candidates or add missing steps; a late superseded result retains its original input provenance and cannot overwrite the replanned evaluation or review draft.

## Knowledge maintenance and next verification

Prioritize actual evidence, propagation/risk, candidate maneuvers and second screening for the conjunction demo, then expand contact/telemetry integration. Track knowledge ingestion, adapter execution verification, current data/health/permissions and workflow planning as separate evidence levels. Never assume commercial access or flight connectivity for a demo.

Preserve tool IDs and source URLs when translating or revising entries. Record changes to capabilities, source versions and verification dates explicitly. First integration path, usable account/data access, capability contracts and equivalence criteria remain open. Internal knowledge status does not grant runtime execution authority or claim deployment.
