# ArgusOrb

**Real-time tracking and anomaly detection for the Starlink constellation.**

ArgusOrb ingests public TLE data, propagates the full Starlink constellation (~10,000+ satellites) with SGP4, persists every orbital update, and visualizes the result in 3D — a foundation for open space situational awareness.

> *"Don't trust the numbers. Verify the physics."*

🛰 **Live:** https://argusorb.io

---

## Features

- **Full-constellation propagation** — every Starlink satellite, updated every few seconds, on a single VPS.
- **Historical TLE archive** — every Celestrak update is persisted to SQLite. No update is ever discarded.
- **Orbital anomaly detection** — flag maneuvers, decay events, and state discontinuities on top of a persisted history.
- **3D visualization** — CesiumJS `PointPrimitiveCollection` rendering, capable of 10,000+ satellites without frame drops.
- **Live telemetry stream** — REST for snapshots, WebSocket for real-time position updates.
- **Orbital shell breakdown** — satellites grouped by altitude shell (340 / 550 / 570 km, etc.).

## Architecture

```
 Celestrak TLE Feed
        │
        ▼
 TLE Fetcher (every 8 h) ──► SQLite archive
        │                         │
        ▼                         ▼
 SGP4 Propagator (~5 s)    Orbital Analyzer
        │                         │
        └──────────┬──────────────┘
                   ▼
          FastAPI + WebSocket
                   │
                   ▼
              CesiumJS 3D
```

| Layer | Module | Role |
|---|---|---|
| Ingestion | `services/telemetry/tle_fetcher.py` | Pull TLEs from Celestrak on a fixed cadence |
| Storage | `services/telemetry/store.py` | SQLite persistence, full history |
| Propagation | `services/telemetry/propagator.py` | SGP4 for the full constellation |
| Analysis | `services/brain/orbital_analyzer.py` | Anomaly detection + shell grouping |
| Validation | `services/brain/skeptic_agent.py` | Physics-based cross-checks |
| API | `services/api/main.py` | FastAPI REST + WebSocket |
| Frontend | `apps/web/` | CesiumJS 3D viewer |

## Quickstart

```bash
# Backend
uv sync
uv run python -m services.api.runner        # serves on :8000

# Frontend (separate terminal)
cd apps/web
npm install
npm run dev                                   # serves on :5173
```

### Production (Docker)

```bash
docker compose up -d --build
```

## API

| Endpoint | Description |
|---|---|
| `GET /api/starlink/constellation` | Current positions for all tracked satellites |
| `GET /api/starlink/satellite/{norad_id}` | Single-satellite detail (position, velocity, orbit) |
| `GET /api/starlink/anomalies` | Recently detected orbital events |
| `GET /api/starlink/shells` | Satellite count per orbital shell |
| `GET /api/status` | Service health check |
| `WS  /ws/telemetry` | Live telemetry stream |

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, SGP4, SQLite, WebSockets, `uv`
- **Frontend:** CesiumJS, vanilla JavaScript, Vite
- **Infra:** Docker, Cloudflare CDN / TLS, Hostinger VPS
- **Data source:** [Celestrak](https://celestrak.org/) public TLEs

## Project Layout

```
services/
  api/           FastAPI app + runner
  brain/         Orbital analyzer, skeptic agent, gravity model, cross-validator
  telemetry/     TLE fetcher, SGP4 propagator, SQLite store, workers
apps/
  web/           CesiumJS frontend
scripts/         Local + Cloudflare R2 backup scripts
deploy/          VPS setup
tests/           Pytest suite
```

## Data Sources

- **TLE:** [Celestrak](https://celestrak.org/) — public two-line element sets for the Starlink constellation, refreshed periodically.

## Status

ArgusOrb is in active development. The Starlink tracker is live; broader SSA features (multi-constellation ingestion, error-ellipsoid propagation, collision screening) are on the roadmap.

## License

TBD.
