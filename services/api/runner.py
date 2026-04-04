"""Combined runner — starts FastAPI server + telemetry workers in one process.

Usage:
    python -m services.api.runner
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import uvicorn

from services.api.main import (
    app, store, alert_store, validator,
    ingest_telemetry, ingest_alert,
    broadcast_telemetry, broadcast_alert, _ws_clients,
)


async def run_all() -> None:
    """Run API server, issinfo scraper, Horizons worker, cross-validator."""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Patch telemetry_worker to use shared store
    from services.telemetry.telemetry_worker import run_worker
    import services.telemetry.telemetry_worker as tw
    tw.store = store

    # Horizons worker
    from services.telemetry.horizons_worker import run_horizons_worker

    def on_horizons_telemetry(point: dict) -> None:
        ingest_telemetry(point)

        # Feed validator: grab latest issinfo readings from Lethe
        recent = store.latest(20)
        for p in recent:
            if p.get("source", "issinfo") != "jpl_horizons":
                validator.update_issinfo(p)

        # Cross-validate
        result = validator.validate(point)
        if result:
            grade = result.grade
            conf = result.confidence
            marker = {"excellent": "+", "good": "~", "degraded": "!", "suspect": "X"}
            print(
                f"  [{marker.get(grade, '?')}VALIDATE] {grade.upper()} "
                f"(confidence={conf:.1%}) "
                f"vel={result.velocity_pct:.2f}% "
                f"earth={result.earth_dist_pct:.2f}% "
                f"moon={result.moon_dist_pct:.2f}%"
            )

            asyncio.create_task(_broadcast_validation(result.to_dict()))

            if grade == "suspect":
                alert = {
                    "type": "Insight_Alert",
                    "timestamp": result.timestamp,
                    "met": point.get("met", ""),
                    "alert_type": "data_quality",
                    "confidence": conf,
                    "deviation_pct": max(
                        result.velocity_pct,
                        result.earth_dist_pct,
                        result.moon_dist_pct,
                    ),
                    "details": result.details,
                }
                ingest_alert(alert)
                asyncio.create_task(broadcast_alert(alert))
        else:
            print("  [?VALIDATE] No issinfo data available for cross-validation")

    # DSN worker
    from services.telemetry.dsn_worker import run_dsn_worker

    def on_dsn_update(contacts):
        asyncio.create_task(_broadcast_dsn(contacts))

    tasks = [
        asyncio.create_task(server.serve()),
        asyncio.create_task(run_worker(with_skeptic=True, api_mode=True)),
        asyncio.create_task(run_horizons_worker(
            on_telemetry=on_horizons_telemetry,
            poll_interval=60,
        )),
        asyncio.create_task(run_dsn_worker(
            on_update=on_dsn_update,
            poll_interval=10,
        )),
    ]

    await asyncio.gather(*tasks)


async def _broadcast_dsn(contacts: list) -> None:
    if not _ws_clients:
        return
    message = json.dumps({"type": "dsn", "data": contacts})
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def _broadcast_validation(result: dict) -> None:
    if not _ws_clients:
        return
    message = json.dumps({"type": "validation", "data": result})
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


def main() -> None:
    def _shutdown(sig: int, frame: object) -> None:
        print(f"\n[STOP] Signal {sig}. Entries: {store.size}, Alerts: {alert_store.size}")
        print(f"[STOP] Validation stats: {validator.stats}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
