# Argus Internal Tool Knowledge Index

This is the entry point for the English space operations tool prior. Read [evidence and use rules](README.md#evidence-interpretation) before planning. Use [catalog.json](catalog.json) for structured retrieval; it is not an executable tool registry.

## Capability map

| Operations area | Candidate knowledge entries | Selection questions |
| :--- | :--- | :--- |
| Catalog, object metadata and event inputs | [CelesTrak](README.md#celestrak), [Space-Track](README.md#space_track), [ESA DISCOSweb](README.md#discos), [LeoLabs Collision Avoidance](README.md#leolabs), [EU SST services](README.md#eu_sst) | Check data type, object identity, epoch, permissions and catalog coverage. |
| Propagation, estimation, frames and time | [python-sgp4 / SGP4](README.md#sgp4), [Orekit](README.md#orekit), [Tudat / TudatPy](README.md#tudat), [NASA GMAT](README.md#gmat), [NAIF SPICE](README.md#spice), [Ansys ODTK](README.md#odtk), [FreeFlyer](README.md#freeflyer), [Kayhan Dynamics](README.md#kayhan_dynamics) | Match element theory, measurements, force/environment models, uncertainty and reference frames. |
| Collision risk and catalog screening | [Orekit](README.md#orekit), [Kayhan SDK screening service (documented as Pathfinder)](README.md#kayhan_screening), [LeoLabs Collision Avoidance](README.md#leolabs), [EU SST services](README.md#eu_sst) | Distinguish Pc for a known encounter from searching a catalog. Check covariance, HBR and historical coverage. |
| Maneuver, trajectory and spacecraft simulation | [NASA GMAT](README.md#gmat), [Ansys STK / Astrogator](README.md#stk), [Basilisk](README.md#basilisk) | Check convergence, propulsion/attitude/payload constraints and the subsequent screening step. |
| Visibility and ground contacts | [Ansys STK / Astrogator](README.md#stk), [Orekit](README.md#orekit), [NAIF SPICE](README.md#spice), [AWS Ground Station](README.md#aws_ground_station) | Distinguish geometric visibility from available and reserved physical resources. |
| Telemetry, alarms, procedures and commands | [Yamcs](README.md#yamcs), [OpenC3 COSMOS](README.md#openc3) | Check mission configuration and separate read, procedure and command permissions. |
| Debris environment, disposal and re-entry | [ESA DRAMA / MASTER](README.md#drama_master), [EU SST services](README.md#eu_sst) | Distinguish statistical/design analysis from current-event products and verify applicability. |

## Tool ID index

| Stable ID | Knowledge card |
| :--- | :--- |
| `celestrak` | [CelesTrak](README.md#celestrak) |
| `space_track` | [Space-Track](README.md#space_track) |
| `discos` | [ESA DISCOSweb](README.md#discos) |
| `sgp4` | [python-sgp4 / SGP4](README.md#sgp4) |
| `orekit` | [Orekit](README.md#orekit) |
| `tudat` | [Tudat / TudatPy](README.md#tudat) |
| `gmat` | [NASA GMAT](README.md#gmat) |
| `spice` | [NAIF SPICE](README.md#spice) |
| `stk` | [Ansys STK / Astrogator](README.md#stk) |
| `odtk` | [Ansys ODTK](README.md#odtk) |
| `freeflyer` | [FreeFlyer](README.md#freeflyer) |
| `kayhan_dynamics` | [Kayhan Dynamics](README.md#kayhan_dynamics) |
| `kayhan_screening` | [Kayhan SDK screening service (documented as Pathfinder)](README.md#kayhan_screening) |
| `leolabs` | [LeoLabs Collision Avoidance](README.md#leolabs) |
| `eu_sst` | [EU SST services](README.md#eu_sst) |
| `basilisk` | [Basilisk](README.md#basilisk) |
| `yamcs` | [Yamcs](README.md#yamcs) |
| `openc3` | [OpenC3 COSMOS](README.md#openc3) |
| `aws_ground_station` | [AWS Ground Station](README.md#aws_ground_station) |
| `drama_master` | [ESA DRAMA / MASTER](README.md#drama_master) |

## Retrieval boundary

The prior identifies possible capabilities and dependencies. Task-specific requirements and live adapter/data/account checks determine eligibility. Capability tags do not establish interchangeable tools. Records with unknown availability or unverified interfaces require verification before execution.

Knowledge role: Argus internal prior. Language: English. Source check date: 2026-10-04. Coverage: 20 representative tools/services and 39 distinct primary-source URLs. Runtime ingestion and actual calls are not implemented by this documentation update.
