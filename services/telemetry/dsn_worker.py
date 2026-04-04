"""DSN Now Worker — Real-time Deep Space Network tracking status.

Polls NASA's DSN Now XML feed for Artemis II (EM2/Orion) tracking data.
Provides: which antenna is tracking, signal strength, data rate, range, RTLT.

Data source: https://eyes.nasa.gov/dsn/data/dsn.xml
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from services.telemetry.horizons_worker import _get_ssl_context
except ImportError:
    def _get_ssl_context():
        import ssl
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl._create_unverified_context()

DSN_URL = "https://eyes.nasa.gov/dsn/data/dsn.xml"
POLL_INTERVAL = 10  # seconds
ARTEMIS_NAMES = {"EM2", "ORION", "ARTEMIS", "ARTM2", "ART2", "INTEGRITY"}

# Station friendly names
STATIONS = {
    "gdscc": "Goldstone, CA",
    "cdscc": "Canberra, AU",
    "mdscc": "Madrid, ES",
}


@dataclass
class DSNContact:
    """A single DSN dish-spacecraft contact."""
    station: str
    station_name: str
    dish: str
    azimuth: float
    elevation: float
    spacecraft: str
    spacecraft_id: str
    signal_type: str  # "uplink" | "downlink" | "both"
    frequency_mhz: float
    band: str
    data_rate_bps: float
    power_dbm: float
    range_km: float
    rtlt_sec: float  # round-trip light time
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "station": self.station,
            "station_name": self.station_name,
            "dish": self.dish,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "spacecraft": self.spacecraft,
            "spacecraft_id": self.spacecraft_id,
            "signal_type": self.signal_type,
            "frequency_mhz": self.frequency_mhz,
            "band": self.band,
            "data_rate_bps": self.data_rate_bps,
            "power_dbm": self.power_dbm,
            "range_km": self.range_km,
            "rtlt_sec": self.rtlt_sec,
            "timestamp": self.timestamp,
        }


def _safe_float(val, default=0.0):
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


async def fetch_dsn_status() -> list[DSNContact]:
    """Fetch current DSN tracking status from NASA."""
    import urllib.request

    ctx = _get_ssl_context()
    loop = asyncio.get_event_loop()

    try:
        response = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(DSN_URL, timeout=15, context=ctx).read()
        )
    except Exception as e:
        print(f"[DSN] Fetch failed: {e}", file=sys.stderr)
        return []

    return _parse_dsn_xml(response)


def _parse_dsn_xml(xml_bytes: bytes) -> list[DSNContact]:
    """Parse DSN XML and extract Artemis II contacts."""
    contacts = []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return contacts

    # DSN XML: stations and dishes are siblings at root level.
    # Track current station as we iterate.
    current_station = ""
    current_station_name = ""

    for elem in root:
        if elem.tag == "station":
            current_station = (elem.get("name", "")).lower()
            current_station_name = STATIONS.get(current_station, current_station)
            continue

        if elem.tag != "dish":
            continue

        dish = elem
        station_code = current_station
        station_name = current_station_name
        dish_name = dish.get("name", "")
        azimuth = _safe_float(dish.get("azimuthAngle"))
        elevation = _safe_float(dish.get("elevationAngle"))

        # Check targets for Artemis II
        for target in dish.findall("target"):
            sc_name = (target.get("name", "")).upper()
            sc_id = target.get("id", "")

            is_artemis = any(n in sc_name for n in ARTEMIS_NAMES) or sc_id in ("-1024", "-24", "24")
            if not is_artemis:
                continue

            range_km = _safe_float(target.get("downlegRange"))
            rtlt = _safe_float(target.get("rtlt"))

            signal_type = "none"
            freq = 0.0
            band = ""
            data_rate = 0.0
            power = 0.0

            for sig in dish.findall("downSignal"):
                sc = sig.get("spacecraft", "").upper()
                if any(n in sc for n in ARTEMIS_NAMES) and sig.get("active") == "true":
                    signal_type = "downlink"
                    freq = _safe_float(sig.get("frequency"))
                    band = sig.get("band", "")
                    data_rate = _safe_float(sig.get("dataRate"))
                    power = _safe_float(sig.get("power"))

            for sig in dish.findall("upSignal"):
                sc = sig.get("spacecraft", "").upper()
                if any(n in sc for n in ARTEMIS_NAMES) and sig.get("active") == "true":
                    if signal_type == "downlink":
                        signal_type = "both"
                    else:
                        signal_type = "uplink"
                    if freq == 0:
                        freq = _safe_float(sig.get("frequency"))
                        band = sig.get("band", "")

            contacts.append(DSNContact(
                station=station_code,
                station_name=station_name,
                dish=dish_name,
                azimuth=azimuth,
                elevation=elevation,
                spacecraft=sc_name,
                spacecraft_id=sc_id,
                signal_type=signal_type,
                frequency_mhz=freq,
                band=band,
                data_rate_bps=data_rate,
                power_dbm=power,
                range_km=range_km,
                rtlt_sec=rtlt,
                timestamp=time.time(),
            ))

    return contacts


# Shared latest state
_latest_contacts: list[DSNContact] = []


def get_latest_dsn() -> list[dict]:
    """Get latest DSN contacts as dicts."""
    return [c.to_dict() for c in _latest_contacts]


async def run_dsn_worker(
    on_update=None,
    poll_interval: int = POLL_INTERVAL,
) -> None:
    """Poll DSN Now for Artemis II tracking status."""
    global _latest_contacts

    print(f"[DSN] Worker starting (interval={poll_interval}s)")

    cycle = 0
    while True:
        cycle += 1
        contacts = await fetch_dsn_status()

        if contacts:
            _latest_contacts = contacts
            for c in contacts:
                print(
                    f"[DSN][{cycle:04d}] {c.dish}@{c.station_name} → {c.spacecraft} "
                    f"| {c.signal_type} {c.band}-band "
                    f"| range={c.range_km:.0f} km RTLT={c.rtlt_sec:.2f}s "
                    f"| rate={c.data_rate_bps:.0f} bps"
                )
            if on_update:
                on_update([c.to_dict() for c in contacts])
        else:
            if cycle % 30 == 1:  # don't spam if no contacts
                print(f"[DSN][{cycle:04d}] No active Artemis II contacts")

        await asyncio.sleep(poll_interval)


def main() -> None:
    def _shutdown(sig, frame):
        print("\n[DSN] Shutting down.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    asyncio.run(run_dsn_worker())


if __name__ == "__main__":
    main()
