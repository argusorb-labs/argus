"""Telemetry Worker — Real-time Artemis II data ingestion.

Scrapes spacecraft telemetry from issinfo.net/artemis.html using
Playwright (headless), every 5 seconds. Data flows into Lethe KV store
for sub-millisecond point lookups.

Usage:
    cd services/telemetry
    pip install playwright
    playwright install chromium
    python telemetry_worker.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
import time

try:
    from services.telemetry.lethe import Lethe
    from services.telemetry.models import TelemetryPoint
except ImportError:
    from lethe import Lethe
    from models import TelemetryPoint

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARTEMIS_URL = "https://issinfo.net/artemis.html"
POLL_INTERVAL = 5  # seconds
HEADLESS = True

# Shared Lethe instance (in production, this would be a network service)
store = Lethe(max_entries=500_000)

# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


def _parse_float(text: str) -> float:
    """Extract a float from a string like '1.234' or '384,400'."""
    cleaned = re.sub(r"[^\d.\-]", "", text.replace(",", ""))
    return float(cleaned) if cleaned else 0.0


async def scrape_telemetry(page) -> TelemetryPoint | None:  # noqa: ANN001
    """Extract telemetry data from the Artemis tracking page.

    Uses the actual DOM structure: .sp-section containers with
    .sp-label + .sp-val pairs, and .sp-met-value for MET.
    """
    try:
        # Wait for JS to populate values (replace placeholder dashes)
        await page.wait_for_function(
            """() => {
                const vals = document.querySelectorAll('.sp-val');
                return vals.length > 0 && !vals[0].textContent.includes('--');
            }""",
            timeout=10_000,
        )

        # Extract all label-value pairs from the side panel
        pairs = await page.evaluate(
            """() => {
                const sections = document.querySelectorAll('.sp-section');
                const data = {};
                sections.forEach(sec => {
                    const rows = sec.querySelectorAll('.sp-label, .sp-val');
                    let currentLabel = '';
                    rows.forEach(el => {
                        if (el.classList.contains('sp-label')) {
                            currentLabel = el.textContent.trim().toUpperCase();
                        } else if (el.classList.contains('sp-val') && currentLabel) {
                            data[currentLabel] = el.textContent.trim();
                        }
                    });
                });
                // MET is in its own section
                const met = document.querySelector('.sp-met-value');
                if (met) data['MET'] = met.textContent.trim();

                // Phase might be a label-val pair or separate element
                const phase = document.querySelector('.sp-phase-val, .sp-val[data-field="phase"]');
                if (phase) data['PHASE'] = phase.textContent.trim();

                return data;
            }"""
        )

        # Parse extracted values
        velocity = 0.0
        earth_dist = 0.0
        moon_dist = 0.0
        met = "unknown"
        phase = "unknown"

        for label, value in pairs.items():
            label_up = label.upper()
            if "VELOCITY" in label_up or "SPEED" in label_up:
                velocity = _parse_float(value)
            elif "EARTH" in label_up:
                earth_dist = _parse_float(value)
            elif "MOON" in label_up:
                moon_dist = _parse_float(value)
            elif label_up == "MET":
                met = value
            elif "PHASE" in label_up:
                phase = value

        # Fallback: if structured extraction missed values, try regex on rendered text
        if velocity == 0.0 or earth_dist == 0.0:
            text = await page.inner_text(".artemis-side-panel")
            if velocity == 0.0:
                m = re.search(r"([\d.]+)\s*km/s", text)
                if m:
                    velocity = float(m.group(1))
            if earth_dist == 0.0:
                m = re.search(r"Earth\s*([\d,.]+)\s*km", text, re.IGNORECASE)
                if m:
                    earth_dist = _parse_float(m.group(1))
            if moon_dist == 0.0:
                m = re.search(r"Moon\s*([\d,.]+)\s*km", text, re.IGNORECASE)
                if m:
                    moon_dist = _parse_float(m.group(1))

        return TelemetryPoint(
            timestamp=time.time(),
            met=met,
            phase=phase,
            velocity_kms=velocity,
            earth_dist_km=earth_dist,
            moon_dist_km=moon_dist,
        )

    except Exception as e:
        print(f"[WARN] Scrape failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Skeptic Agent integration
# ---------------------------------------------------------------------------


def _run_skeptic(point: TelemetryPoint, agent) -> None:  # noqa: ANN001
    """Run the Skeptic Agent on a telemetry point and print alerts."""
    alert = agent.analyze(point.to_dict())
    if alert:
        print(f"  [ALERT] {alert.alert_type}: {alert.details}")


def _run_skeptic_and_return(point: TelemetryPoint, agent):  # noqa: ANN001, ANN202
    """Run the Skeptic Agent and return the alert (or None)."""
    alert = agent.analyze(point.to_dict())
    if alert:
        print(f"  [ALERT] {alert.alert_type}: {alert.details}")
    return alert


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_worker(
    with_skeptic: bool = True,
    api_mode: bool = False,
) -> None:
    """Main telemetry ingestion loop.

    Args:
        with_skeptic: Enable Skeptic Agent physics verification.
        api_mode: If True, also push data into the API stores and broadcast via WS.
    """
    from playwright.async_api import async_playwright

    # Optional API integration
    _ingest_telemetry = None
    _ingest_alert = None
    _broadcast_telemetry = None
    _broadcast_alert = None
    if api_mode:
        try:
            from services.api.main import (
                ingest_telemetry as _ingest_telemetry,
                ingest_alert as _ingest_alert,
                broadcast_telemetry as _broadcast_telemetry,
                broadcast_alert as _broadcast_alert,
            )
            print("[INIT] API integration enabled")
        except ImportError:
            print("[INIT] API module not available, running standalone")
            api_mode = False

    print(f"[INIT] Telemetry worker starting (interval={POLL_INTERVAL}s)")
    print(f"[INIT] Target: {ARTEMIS_URL}")
    print(f"[INIT] Headless: {HEADLESS}")

    # Initialize Skeptic Agent
    skeptic = None
    if with_skeptic:
        try:
            from services.brain.skeptic_agent import SkepticAgent
            skeptic = SkepticAgent(anomaly_threshold_pct=0.5)
            print("[INIT] Skeptic Agent loaded")
        except ImportError:
            try:
                sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "brain"))
                from skeptic_agent import SkepticAgent
                skeptic = SkepticAgent(anomaly_threshold_pct=0.5)
                print("[INIT] Skeptic Agent loaded (via sys.path)")
            except ImportError:
                print("[INIT] Skeptic Agent not available (run from monorepo root)")

    async with async_playwright() as p:
        chromium_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
        browser = await p.chromium.launch(
            headless=HEADLESS,
            executable_path=chromium_path,
        )
        page = await browser.new_page()

        print("[INIT] Loading Artemis tracking page...")
        try:
            await page.goto(ARTEMIS_URL, wait_until="networkidle", timeout=30_000)
        except Exception:
            # networkidle may timeout on heavy pages — domcontentloaded is enough
            await page.goto(ARTEMIS_URL, wait_until="domcontentloaded", timeout=30_000)

        # Give JS time to hydrate
        await asyncio.sleep(3)
        print("[INIT] Page loaded. Starting telemetry collection.\n")

        cycle = 0
        while True:
            cycle += 1
            t0 = time.perf_counter()

            point = await scrape_telemetry(page)

            if point and (point.velocity_kms > 0 or point.earth_dist_km > 0):
                # Write to Lethe
                store.put(point.key, point.to_dict(), timestamp=point.timestamp)
                store.put(point.timeseries_key, point.to_dict())

                elapsed_ms = (time.perf_counter() - t0) * 1000

                print(
                    f"[{cycle:05d}] MET={point.met}  "
                    f"v={point.velocity_kms:.3f} km/s  "
                    f"Earth={point.earth_dist_km:.0f} km  "
                    f"Moon={point.moon_dist_km:.0f} km  "
                    f"Phase={point.phase}  "
                    f"({elapsed_ms:.1f}ms, store={store.size})"
                )

                # Push to API stores + broadcast
                point_dict = point.to_dict()
                if api_mode and _ingest_telemetry:
                    _ingest_telemetry(point_dict)
                if api_mode and _broadcast_telemetry:
                    asyncio.create_task(_broadcast_telemetry(point_dict))

                # Run physics verification
                if skeptic:
                    alert = _run_skeptic_and_return(point, skeptic)
                    if alert and api_mode and _ingest_alert:
                        alert_dict = json.loads(alert.to_json())
                        _ingest_alert(alert_dict)
                        if _broadcast_alert:
                            asyncio.create_task(_broadcast_alert(alert_dict))
            else:
                print(f"[{cycle:05d}] No valid data (page may still be loading)")

            # Benchmark every 100 cycles
            if cycle % 100 == 0 and store.size > 0:
                latest = store.latest(1)
                if latest:
                    sample_key = f"telem:{latest[0]['met']}"
                    latency_us = store.bench_point_lookup(sample_key)
                    print(
                        f"        [BENCH] Point lookup: {latency_us:.1f} us "
                        f"({latency_us / 1000:.3f} ms) | "
                        f"Store: {store.size} entries"
                    )
                if skeptic:
                    print(f"        [SKEPTIC] {skeptic.stats}")

            await asyncio.sleep(POLL_INTERVAL)


def main() -> None:
    """Entry point with graceful shutdown."""

    def _shutdown(sig: int, frame: object) -> None:
        print(f"\n[STOP] Signal {sig} received.")
        print(f"[STOP] Total writes: {store.total_writes}, store size: {store.size}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        print(f"\n[STOP] Total writes: {store.total_writes}")


if __name__ == "__main__":
    main()
