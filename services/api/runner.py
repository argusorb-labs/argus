"""Combined runner — starts FastAPI server + telemetry workers in one process.

Usage:
    python -m services.api.runner
"""

from __future__ import annotations

import asyncio
import signal
import sys
import uvicorn

from services.api.main import app, store, alert_store, ingest_telemetry, broadcast_telemetry


async def run_all() -> None:
    """Run API server, issinfo scraper, and Horizons worker concurrently."""
    # Start uvicorn as an async server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Start telemetry worker (issinfo.net scraper)
    from services.telemetry.telemetry_worker import run_worker
    import services.telemetry.telemetry_worker as tw
    tw.store = store

    # Start Horizons worker (JPL ephemeris)
    from services.telemetry.horizons_worker import run_horizons_worker

    def on_horizons_telemetry(point: dict) -> None:
        """Ingest Horizons data into the shared store."""
        ingest_telemetry(point)

    tasks = [
        asyncio.create_task(server.serve()),
        asyncio.create_task(run_worker(with_skeptic=True, api_mode=True)),
        asyncio.create_task(run_horizons_worker(
            on_telemetry=on_horizons_telemetry,
            poll_interval=60,
        )),
    ]

    await asyncio.gather(*tasks)


def main() -> None:
    def _shutdown(sig: int, frame: object) -> None:
        print(f"\n[STOP] Signal {sig}. Entries: {store.size}, Alerts: {alert_store.size}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
