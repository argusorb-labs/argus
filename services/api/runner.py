"""Combined runner — FastAPI + TLE fetcher + position updater + anomaly analyzer.

Usage:
    python -m services.api.runner
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
import uvicorn

from services.api.main import (
    app, store, propagator, update_position_cache, broadcast,
)
from services.telemetry.tle_fetcher import run_tle_fetcher
from services.telemetry.supgp_fetcher import run_supgp_fetcher
from services.brain.orbital_analyzer import analyze_constellation


async def position_update_loop(interval: int = 5) -> None:
    """Refresh position cache every N seconds and broadcast to WS clients."""
    print(f"[POS] Position updater starting (interval={interval}s)")
    while True:
        t0 = time.perf_counter()
        update_position_cache()
        elapsed = time.perf_counter() - t0

        # Broadcast positions to WebSocket clients
        from services.api.main import _position_cache
        await broadcast("positions", {
            "count": _position_cache["count"],
            "timestamp": _position_cache["timestamp"],
        })

        await asyncio.sleep(interval)


def on_tle_fetch_complete(total: int, new: int) -> None:
    """Called after each TLE fetch. Reload propagator + run anomaly detection."""
    # Reload propagator with fresh TLEs
    tles = store.get_latest_tles()
    loaded = propagator.load_tles(tles)
    print(f"[TLE] Propagator reloaded: {loaded} satellites")

    # Run anomaly detection
    anomalies = analyze_constellation(store)
    if anomalies:
        print(f"[ANOMALY] Detected {len(anomalies)} anomalies:")
        for a in anomalies[:5]:
            print(f"  {a['anomaly_type']}: norad={a['norad_id']} {a.get('details','')}")
        # Broadcast anomalies (fire-and-forget in the event loop)
        loop = asyncio.get_event_loop()
        for a in anomalies:
            loop.create_task(broadcast("anomaly", a))


async def run_all() -> None:
    """Start all services."""
    # Initial TLE load
    print("[INIT] Loading TLEs from database...")
    tles = store.get_latest_tles()
    if tles:
        loaded = propagator.load_tles(tles)
        update_position_cache()
        print(f"[INIT] Loaded {loaded} satellites from database")
    else:
        print("[INIT] No TLEs in database, will fetch on first cycle")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    tasks = [
        asyncio.create_task(server.serve()),
        asyncio.create_task(run_tle_fetcher(
            store=store,
            on_complete=on_tle_fetch_complete,
            interval=8 * 3600,
        )),
        asyncio.create_task(run_supgp_fetcher(
            store=store,
            interval=8 * 3600,
        )),
        asyncio.create_task(position_update_loop(interval=5)),
    ]

    await asyncio.gather(*tasks)


def main() -> None:
    def _shutdown(sig, frame):
        stats = store.stats
        print(f"\n[STOP] Signal {sig}. {stats}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
