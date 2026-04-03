"""Selene-Insight API — REST + WebSocket gateway to telemetry and alerts.

Exposes Lethe KV store and Skeptic Agent alerts over HTTP/WS.

Usage:
    uvicorn services.api.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from services.telemetry.lethe import Lethe
from services.brain.cross_validator import CrossValidator

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

store = Lethe(max_entries=500_000)
alert_store = Lethe(max_entries=10_000)
validator = CrossValidator(time_tolerance_sec=120)

# Connected WebSocket clients
_ws_clients: set[WebSocket] = set()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown lifecycle."""
    print("[API] Selene-Insight API started")
    yield
    print("[API] Selene-Insight API shutting down")


app = FastAPI(
    title="Selene-Insight API",
    description="Artemis II real-time telemetry and physics verification",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/telemetry/latest")
async def telemetry_latest(n: int = Query(default=10, ge=1, le=1000)):
    """Get the N most recent telemetry readings."""
    return {"data": store.latest(n), "count": store.size}


@app.get("/api/telemetry/range")
async def telemetry_range(
    start: float = Query(..., description="Start timestamp (Unix epoch)"),
    end: float = Query(..., description="End timestamp (Unix epoch)"),
    limit: int = Query(default=1000, ge=1, le=10000),
):
    """Query telemetry by time range (for dashboard rewind)."""
    data = store.range(start, end, limit=limit)
    return {"data": data, "count": len(data)}


@app.get("/api/alerts/latest")
async def alerts_latest(n: int = Query(default=20, ge=1, le=500)):
    """Get the N most recent Skeptic Agent alerts."""
    return {"data": alert_store.latest(n), "count": alert_store.size}


@app.get("/api/validation/latest")
async def validation_latest():
    """Latest cross-validation results and confidence score."""
    return {
        "stats": validator.stats,
        "recent": validator.recent_results,
    }


@app.get("/api/status")
async def status():
    """Health check with store stats."""
    v = validator.stats
    return {
        "status": "ok",
        "telemetry_entries": store.size,
        "alert_entries": alert_store.size,
        "ws_clients": len(_ws_clients),
        "data_confidence": v.get("latest_confidence"),
        "data_grade": v.get("latest_grade"),
        "validations": v.get("total_validations"),
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    """Live telemetry stream. Pushes new readings as they arrive."""
    await ws.accept()
    _ws_clients.add(ws)
    try:
        # Keep connection alive; client can send pings
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


async def broadcast_telemetry(point: dict) -> None:
    """Push a telemetry point to all connected WebSocket clients."""
    if not _ws_clients:
        return
    message = json.dumps({"type": "telemetry", "data": point})
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def broadcast_alert(alert: dict) -> None:
    """Push a Skeptic Agent alert to all connected WebSocket clients."""
    if not _ws_clients:
        return
    message = json.dumps({"type": "alert", "data": alert})
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Ingestion helpers (called by telemetry worker)
# ---------------------------------------------------------------------------

def ingest_telemetry(point: dict) -> None:
    """Store a telemetry point and queue broadcast."""
    ts = point.get("timestamp", time.time())
    met = point.get("met", "unknown")
    source = point.get("source", "issinfo")
    store.put(f"telem:{source}:{met}", point, timestamp=ts)


def ingest_alert(alert: dict) -> None:
    """Store a Skeptic Agent alert and queue broadcast."""
    ts = alert.get("timestamp", time.time())
    met = alert.get("met", "unknown")
    alert_store.put(f"alert:{met}", alert, timestamp=ts)
