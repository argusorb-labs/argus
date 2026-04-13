"""Selene-Insight API — Starlink constellation tracking.

REST + WebSocket gateway for satellite positions, anomalies, and metadata.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Path
from fastapi.middleware.cors import CORSMiddleware

from services.telemetry.store import StarlinkStore
from services.telemetry.propagator import Propagator

# ── Shared state ──

store = StarlinkStore()
propagator = Propagator()
_ws_clients: set[WebSocket] = set()

# Position cache (refreshed every 5s by runner)
_position_cache: dict = {"satellites": [], "timestamp": 0, "count": 0}


def update_position_cache() -> None:
    """Recompute all satellite positions. Called by runner every 5s."""
    global _position_cache
    positions = propagator.propagate_all()
    _position_cache = {
        "satellites": positions,
        "timestamp": time.time(),
        "count": len(positions),
    }


# ── App ──

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("[API] Selene-Insight Starlink API started")
    yield
    print("[API] Shutting down")


app = FastAPI(
    title="Selene-Insight API",
    description="Starlink constellation tracking and space situational awareness",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──

@app.get("/api/starlink/constellation")
async def constellation(time_unix: float | None = Query(default=None)):
    """All satellite positions. Uses cache for current time, propagates for custom time."""
    if time_unix is not None:
        positions = propagator.propagate_all(time_unix)
        return {"satellites": positions, "timestamp": time_unix, "count": len(positions)}
    return _position_cache


@app.get("/api/starlink/satellite/{norad_id}")
async def satellite_detail(norad_id: int = Path(...)):
    """Single satellite metadata + TLE history."""
    sat = store.get_satellite(norad_id)
    if not sat:
        return {"error": "Satellite not found"}

    history = store.get_satellite_history(norad_id, limit=50)

    # Current position
    from services.telemetry.propagator import propagate_single, tle_to_satrec
    pos = None
    if history:
        satrec = tle_to_satrec(history[0]["line1"], history[0]["line2"])
        if satrec:
            pos = propagate_single(satrec, time.time())

    return {
        "satellite": dict(sat),
        "position": pos,
        "tle_history": [{"epoch_jd": h["epoch_jd"], "mean_motion": h["mean_motion"],
                         "inclination": h["inclination"], "eccentricity": h["eccentricity"]}
                        for h in history],
        "tle_count": len(history),
    }


@app.get("/api/starlink/anomalies")
async def anomalies(limit: int = Query(default=50, ge=1, le=500)):
    """Recent orbital anomalies."""
    return {"anomalies": store.get_anomalies(limit)}


@app.get("/api/starlink/shells")
async def shells():
    """Satellite count by orbital shell."""
    sats = _position_cache.get("satellites", [])
    shell_counts = {}
    for s in sats:
        k = s.get("shell_km", 0)
        shell_counts[k] = shell_counts.get(k, 0) + 1
    return {"shells": dict(sorted(shell_counts.items())), "total": len(sats)}


@app.get("/api/status")
async def status():
    """Health check."""
    stats = store.stats
    return {
        "status": "ok",
        "satellites": stats["satellites"],
        "tle_records": stats["tle_records"],
        "anomalies": stats["anomalies"],
        "position_cache_age_sec": round(time.time() - _position_cache.get("timestamp", 0), 1),
        "ws_clients": len(_ws_clients),
    }


# ── WebSocket ──

@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


async def broadcast(msg_type: str, data) -> None:
    """Push a message to all WebSocket clients."""
    if not _ws_clients:
        return
    message = json.dumps({"type": msg_type, "data": data})
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)
