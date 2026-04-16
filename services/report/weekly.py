"""Weekly space congestion report — builder, renderers, CLI.

The report is the GTM-idea-G deliverable and the gating milestone for
open-sourcing the argus repo. Its strategic job is to convince readers
(operators, insurers, researchers) that we see the Starlink constellation
more clearly than they can from public TLEs alone. Tone is technical and
hedged — every flagged event is a "candidate", never a "confirmed"
anything.

Layering:
    build_report(store, start_ts, end_ts) -> dict   # pure data
    render_markdown(report, previous, editor_notes) -> str
    render_json(report) -> str
    CLI: python -m services.report.weekly [--week YYYY-Www | --ending YYYY-MM-DD]

The CLI does NOT publish anything. It writes files to disk and prints the
path. Publishing (email, blog, etc.) is deliberately out of scope.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from services.telemetry.store import StarlinkStore
except ImportError:
    from telemetry.store import StarlinkStore  # type: ignore

REPORT_SCHEMA_VERSION = 1
AUTHORITATIVE_CLASSIFIER = "rule_v1"
STALE_THRESHOLD_S = 7 * 86400       # "departed" = no TLE in 7 days
CONSTELLATION_FRESHNESS_S = 86400   # "currently tracked" = last_seen within 24h
TOP_FLAGS_FOR_EDITOR = 8            # how many high-confidence flags to surface
DEFAULT_REPORT_DIR = Path(os.environ.get("ARGUS_REPORTS_DIR", "reports"))
SHELL_MIN_POPULATION = 100          # shells with fewer sats than this get bucketed into "other"
MAX_LIST_ITEMS = 30                 # truncate new/departed/flagged lists beyond this in markdown
MAX_PLAUSIBLE_NEW_PER_WEEK = 200    # if "new" count exceeds this, it's a data reset artifact, not real launches

# Human-friendly ordering for the flagged-events table.
CAUSE_ORDER = ["maneuver_candidate", "atmospheric_anomaly", "natural_decay", "reentry"]
ANOMALY_TYPE_ORDER = [
    "altitude_change",
    "inclination_shift",
    "eccentricity_change",
    "deorbiting",
    "reentry",
]


# ── Time windowing ──

def iso_week_bounds(year: int, week: int) -> tuple[float, float]:
    """Return (start_ts, end_ts) UTC timestamps for the ISO week.

    start = Monday 00:00:00 UTC, end = following Monday 00:00:00 UTC
    (half-open interval, so membership is `start <= t < end`).
    """
    start = datetime.fromisocalendar(year, week, 1).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=7)
    return start.timestamp(), end.timestamp()


def parse_week_string(s: str) -> tuple[int, int]:
    """Parse 'YYYY-Www' into (year, week)."""
    year_str, week_str = s.split("-W")
    return int(year_str), int(week_str)


def most_recent_complete_week(now: datetime | None = None) -> tuple[int, int]:
    """ISO (year, week) of the most recently COMPLETED week.

    Seven days ago is always in the most recently completed week, regardless
    of whether today is Monday (just rolled over) or Sunday (current week
    hasn't ended yet).
    """
    now = now or datetime.now(timezone.utc)
    ref = now - timedelta(days=7)
    iso = ref.isocalendar()
    return iso[0], iso[1]


def _format_window_label(start_ts: float, end_ts: float) -> str:
    start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    # end is half-open, so the last full day is end - 1 day
    last = datetime.fromtimestamp(end_ts - 1, tz=timezone.utc)
    return f"{start.strftime('%Y-%m-%d')} → {last.strftime('%Y-%m-%d')}"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes:02d}m"


def _shell_key(shell_km: float) -> str:
    """Stringify a shell for JSON-safe dict keys."""
    if shell_km is None:
        return "unknown"
    if shell_km <= 0:
        return "decayed"
    return f"{int(round(shell_km))}"


# ── Report builder ──

def build_report(
    store: StarlinkStore,
    start_ts: float,
    end_ts: float,
    *,
    classified_by: str = AUTHORITATIVE_CLASSIFIER,
    iso_week: str | None = None,
) -> dict[str, Any]:
    """Query the store and return a structured report dict.

    Pure with respect to the store at call time: same inputs + same DB state
    produce the same dict, so tests and render calls can inspect it directly.
    """
    # ── Constellation population (currently tracked) ──
    # Clamp to "now" so partial-week reports don't treat all sats as stale
    # (end_ts for an incomplete week is in the future, but last_seen is ≤ now).
    effective_end = min(end_ts, time.time())
    shell_counts_raw = store.count_fresh_by_shell(
        as_of_ts=effective_end, freshness_s=CONSTELLATION_FRESHNESS_S
    )
    # Consolidate small shells into "other" to keep the report table clean.
    shell_counts: dict[str, int] = {}
    other_count = 0
    for km, n in shell_counts_raw.items():
        key = _shell_key(km)
        if n >= SHELL_MIN_POPULATION or key in ("decayed", "unknown"):
            shell_counts[key] = n
        else:
            other_count += n
    if other_count > 0:
        shell_counts["other"] = other_count
    total_tracked = sum(shell_counts.values())

    # ── New / departed satellites ──
    new_sats_raw = store.get_new_satellites(since_ts=start_ts)
    new_sats_all = [
        {
            "norad_id": s["norad_id"],
            "name": s.get("name"),
            "first_seen_ts": s.get("first_seen"),
            "shell_km": s.get("shell_km"),
            "launch_group": s.get("launch_group"),
            "intl_designator": s.get("intl_designator"),
        }
        for s in new_sats_raw
        if (s.get("first_seen") or 0) < effective_end
    ]
    # Sanity check: if "new" count exceeds what's physically possible in a
    # week, it's a data coverage expansion (e.g., DB reset), not real launches.
    if len(new_sats_all) > MAX_PLAUSIBLE_NEW_PER_WEEK:
        new_sats = []  # suppress the artifact; the report notes the omission
        new_sats_note = (
            f"{len(new_sats_all)} satellites appeared as 'new' — likely a data "
            f"coverage expansion rather than actual launches. Suppressed."
        )
    else:
        new_sats = new_sats_all
        new_sats_note = None

    departed_sats_raw = store.get_stale_satellites(
        max_age_s=STALE_THRESHOLD_S, now_ts=effective_end
    )
    departed_sats = [
        {
            "norad_id": s["norad_id"],
            "name": s.get("name"),
            "last_seen_ts": s.get("last_seen"),
            "shell_km": s.get("shell_km"),
        }
        for s in departed_sats_raw
    ]

    # ── Flagged events (labels from the authoritative classifier only) ──
    labels = store.get_anomalies_in_window(
        start_ts, end_ts, classified_by=classified_by
    )
    by_cause = Counter(l.get("cause") or "unknown" for l in labels)
    by_type = Counter(l.get("anomaly_type") or "unknown" for l in labels)

    # Top-N by confidence for the editor to pick from. Ties broken by
    # detected_at (newer first) for determinism.
    def _sort_key(label: dict) -> tuple:
        return (-(label.get("confidence") or 0), -(label.get("detected_at") or 0))

    top_flags = [
        {
            "norad_id": l["norad_id"],
            "name": l.get("name"),
            "anomaly_type": l.get("anomaly_type"),
            "cause": l.get("cause"),
            "confidence": l.get("confidence"),
            "details": l.get("details"),
            "detected_at": l.get("detected_at"),
            "altitude_before_km": l.get("altitude_before_km"),
            "altitude_after_km": l.get("altitude_after_km"),
        }
        for l in sorted(labels, key=_sort_key)[:TOP_FLAGS_FOR_EDITOR]
    ]

    # ── Data quality ──
    fetches = store.get_fetch_log_in_window(start_ts, end_ts)
    fetch_attempts = len(fetches)
    fetch_successes = sum(1 for f in fetches if f.get("status") == "ok")
    parse_errors = sum(f.get("parse_errors") or 0 for f in fetches)

    if len(fetches) >= 2:
        gaps = [
            fetches[i + 1]["fetched_at"] - fetches[i]["fetched_at"]
            for i in range(len(fetches) - 1)
        ]
        longest_gap_s = max(gaps)
    elif len(fetches) == 1:
        longest_gap_s = 0
    else:
        longest_gap_s = (end_ts - start_ts)  # full window = no data

    data_quality = {
        "fetch_attempts": fetch_attempts,
        "fetch_successes": fetch_successes,
        "parse_errors": parse_errors,
        "longest_gap_seconds": longest_gap_s,
        "longest_gap_human": _format_duration(longest_gap_s),
    }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": time.time(),
        "classified_by": classified_by,
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "iso_week": iso_week,
            "label": _format_window_label(start_ts, end_ts),
        },
        "constellation": {
            "total": total_tracked,
            "shells": shell_counts,
        },
        "new_satellites": new_sats,
        "new_satellites_note": new_sats_note,
        "departed_satellites": departed_sats,
        "flagged_events": {
            "total": len(labels),
            "by_cause": dict(by_cause),
            "by_anomaly_type": dict(by_type),
            "top_by_confidence": top_flags,
        },
        "data_quality": data_quality,
    }


# ── Delta computation ──

def _delta(current: int | float | None, previous: int | float | None) -> int | float | None:
    if current is None or previous is None:
        return None
    return current - previous


def compute_deltas(current: dict, previous: dict | None) -> dict[str, Any]:
    """Return a dict of week-over-week deltas relative to `previous`.

    Returns an empty dict if previous is None or schema-incompatible.
    """
    if previous is None or previous.get("schema_version") != REPORT_SCHEMA_VERSION:
        return {}

    prev_const = previous.get("constellation", {})
    prev_shells = prev_const.get("shells", {})
    prev_flags = previous.get("flagged_events", {})
    prev_by_cause = prev_flags.get("by_cause", {})
    prev_by_type = prev_flags.get("by_anomaly_type", {})

    cur_const = current.get("constellation", {})
    cur_shells = cur_const.get("shells", {})
    cur_flags = current.get("flagged_events", {})
    cur_by_cause = cur_flags.get("by_cause", {})
    cur_by_type = cur_flags.get("by_anomaly_type", {})

    shell_keys = set(cur_shells) | set(prev_shells)

    return {
        "constellation_total": _delta(cur_const.get("total"), prev_const.get("total")),
        "shells": {
            k: _delta(cur_shells.get(k, 0), prev_shells.get(k, 0)) for k in shell_keys
        },
        "by_cause": {
            k: _delta(cur_by_cause.get(k, 0), prev_by_cause.get(k, 0))
            for k in set(cur_by_cause) | set(prev_by_cause)
        },
        "by_anomaly_type": {
            k: _delta(cur_by_type.get(k, 0), prev_by_type.get(k, 0))
            for k in set(cur_by_type) | set(prev_by_type)
        },
    }


# ── Renderers ──

def _fmt_delta(d: int | float | None) -> str:
    if d is None:
        return "—"
    if d == 0:
        return "0"
    return f"{d:+d}" if isinstance(d, int) else f"{d:+.1f}"


def render_markdown(
    report: dict,
    previous: dict | None = None,
    editor_notes: str | None = None,
) -> str:
    """Render the report as a markdown document for human consumption.

    Args:
        report: a dict from build_report
        previous: optional previous-week report for delta columns
        editor_notes: optional hand-written markdown block that replaces the
            auto-generated "Notable flags" section. This is where you add
            the human-written editorial observation each week.
    """
    r = report
    deltas = compute_deltas(r, previous)
    window = r["window"]
    iso_week = window.get("iso_week") or ""
    title = f"ArgusOrb Weekly — {iso_week}" if iso_week else "ArgusOrb Weekly"

    const = r["constellation"]
    fetches = r["data_quality"]
    flags = r["flagged_events"]
    shells = const.get("shells", {})
    shells_delta = deltas.get("shells", {})

    shell_rows: list[str] = []
    for key in sorted(shells.keys(), key=lambda k: (k != "decayed", k)):
        count = shells[key]
        if key in ("decayed", "unknown", "other"):
            label = key
        else:
            label = f"{key} km shell"
        d = _fmt_delta(shells_delta.get(key))
        shell_rows.append(f"| {label} | {count:,} | {d} |")

    constellation_delta = _fmt_delta(deltas.get("constellation_total"))

    cause_rows: list[str] = []
    cause_counts = flags.get("by_cause", {})
    cause_delta = deltas.get("by_cause", {})
    ordered_causes = [c for c in CAUSE_ORDER if c in cause_counts] + [
        c for c in cause_counts if c not in CAUSE_ORDER
    ]
    for cause in ordered_causes:
        cause_rows.append(
            f"| {cause.replace('_', ' ')} | {cause_counts[cause]} | {_fmt_delta(cause_delta.get(cause))} |"
        )

    # Editor / notable flags section
    if editor_notes is not None and editor_notes.strip():
        notable_section = editor_notes.strip()
    else:
        notable_section = _render_auto_notable(flags.get("top_by_confidence", []))

    # New / departed tables
    new_rows = [
        f"| {s['norad_id']} | {s.get('name') or '—'} | "
        f"{_fmt_ts(s.get('first_seen_ts'))} | {_shell_km_label(s.get('shell_km'))} |"
        for s in r["new_satellites"]
    ]
    departed_rows = [
        f"| {s['norad_id']} | {s.get('name') or '—'} | "
        f"{_fmt_ts(s.get('last_seen_ts'))} | {_shell_km_label(s.get('shell_km'))} |"
        for s in r["departed_satellites"]
    ]

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Reporting window:** {window['label']} UTC  ")
    lines.append(
        f"**Data source:** Celestrak `sup-gp` (Starlink), "
        f"{fetches['fetch_attempts']} fetches over the week  "
    )
    lines.append(f"**Classifier:** `{r['classified_by']}` (threshold-based, unaudited)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Constellation
    lines.append("## Constellation")
    lines.append("")
    lines.append("| | Count | Δ vs prev week |")
    lines.append("|---|---:|---:|")
    lines.append(f"| **Tracked Starlink** | **{const['total']:,}** | {constellation_delta} |")
    lines.extend(shell_rows)
    lines.append("")

    # New
    lines.append("## New to orbit")
    lines.append("")
    new_note = r.get("new_satellites_note")
    if new_note:
        lines.append(f"*{new_note}*")
    elif new_rows:
        noun = "satellite" if len(new_rows) == 1 else "satellites"
        lines.append(f"**{len(new_rows)} {noun}** first appeared in Celestrak during the window.")
        lines.append("")
        lines.append("| NORAD | Name | First seen | Shell |")
        lines.append("|---:|---|---|---:|")
        if len(new_rows) > MAX_LIST_ITEMS:
            lines.extend(new_rows[:MAX_LIST_ITEMS])
            lines.append("")
            lines.append(f"*… and {len(new_rows) - MAX_LIST_ITEMS} more (truncated).*")
        else:
            lines.extend(new_rows)
    else:
        lines.append("No new satellites appeared this week.")
    lines.append("")

    # Departed
    lines.append("## Departed")
    lines.append("")
    if departed_rows:
        noun = "satellite" if len(departed_rows) == 1 else "satellites"
        has = "has" if len(departed_rows) == 1 else "have"
        lines.append(
            f"**{len(departed_rows)} {noun}** {has} not produced a TLE in more than "
            f"{STALE_THRESHOLD_S // 86400} days. Likely deorbited, decommissioned, or renamed — "
            "no Celestrak decay notice has been issued for any of them yet."
        )
        lines.append("")
        lines.append("| NORAD | Name | Last seen | Last shell |")
        lines.append("|---:|---|---|---:|")
        if len(departed_rows) > MAX_LIST_ITEMS:
            lines.extend(departed_rows[:MAX_LIST_ITEMS])
            lines.append("")
            lines.append(f"*… and {len(departed_rows) - MAX_LIST_ITEMS} more (truncated).*")
        else:
            lines.extend(departed_rows)
    else:
        lines.append("No satellites have gone silent this week.")
    lines.append("")

    # Flagged events
    lines.append(f"## Flagged events (`{r['classified_by']}`)")
    lines.append("")
    lines.append(
        f"`{r['classified_by']}` is a threshold classifier. Every flag below is a "
        "**candidate**, not a verified event — the upcoming IMM-UKF classifier "
        "and human review will override these labels over time."
    )
    lines.append("")
    if cause_rows:
        lines.append("| Cause | This week | vs prev |")
        lines.append("|---|---:|---:|")
        lines.extend(cause_rows)
    else:
        lines.append("No events flagged this week.")
    lines.append("")
    lines.append("**Notable flags**")
    lines.append("")
    lines.append(notable_section)
    lines.append("")

    # Data quality
    lines.append("## Data quality")
    lines.append("")
    lines.append("Transparency disclosure — we publish our own fetch audit so readers can discount our claims accordingly.")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---:|")
    lines.append(f"| Fetch attempts | {fetches['fetch_attempts']} |")
    lines.append(f"| Successful | {fetches['fetch_successes']} |")
    lines.append(f"| Parse errors | {fetches['parse_errors']} |")
    lines.append(f"| Longest fetch gap | {fetches['longest_gap_human']} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"*ArgusOrb ingests public Celestrak TLEs, persists every update to a SQLite "
        f"archive, and runs `{r['classified_by']}` against consecutive TLEs. Every flagged "
        f"event above is one opinion from one classifier and will be reviewed by human "
        f"and IMM-UKF tracks. Raw archive and fetch audit are available on request.*"
    )

    return "\n".join(lines) + "\n"


def _render_auto_notable(top_flags: list[dict]) -> str:
    """Auto-render the top-3 highest-confidence flags as bullets.

    This is the fallback when no editor_notes are provided. Mechanical — the
    weekly workflow should prefer hand-written editor notes since this is the
    most strategically valuable section of the report.
    """
    if not top_flags:
        return "_No high-confidence flags this week._"
    lines = []
    for f in top_flags[:3]:
        conf = f.get("confidence")
        conf_str = f"{conf:.2f}" if conf is not None else "—"
        name = f.get("name") or "—"
        norad = f["norad_id"]
        details = f.get("details") or ""
        lines.append(
            f"- **{name}** ({norad}) — `{f.get('cause')}` "
            f"(conf {conf_str}). {details}"
        )
    lines.append("")
    lines.append("_Auto-generated from top-N by confidence. Prefer hand-written editor notes for the final report._")
    return "\n".join(lines)


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _shell_km_label(shell_km: float | None) -> str:
    if shell_km is None:
        return "—"
    if shell_km <= 0:
        return "decayed"
    return f"{int(round(shell_km))} km"


def render_json(report: dict) -> str:
    """JSON rendering for persistence and programmatic consumption."""
    return json.dumps(report, indent=2, sort_keys=True, default=str)


# ── Persistence ──

def load_previous_report(reports_dir: Path, iso_week: str) -> dict | None:
    """Load the report JSON for the week immediately preceding iso_week, if present."""
    try:
        year_str, week_str = iso_week.split("-W")
        year, week = int(year_str), int(week_str)
    except (ValueError, AttributeError):
        return None
    # Go back one ISO week; datetime handles year rollovers.
    prev_start = datetime.fromisocalendar(year, week, 1) - timedelta(days=7)
    prev_iso = prev_start.isocalendar()
    prev_tag = f"{prev_iso[0]}-W{prev_iso[1]:02d}"
    path = reports_dir / f"{prev_tag}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# ── CLI ──

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m services.report.weekly",
        description="Generate the ArgusOrb weekly report from the current store.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--week", help="ISO week to report, e.g. 2026-W15")
    g.add_argument("--ending", help="UTC date (YYYY-MM-DD) inside the target week")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"Where to write reports (default: {DEFAULT_REPORT_DIR}, "
             f"override with ARGUS_REPORTS_DIR env var)",
    )
    p.add_argument(
        "--editor-notes",
        type=Path,
        help="Path to a markdown file that replaces the auto-generated "
             "'Notable flags' section. Recommended for every published report.",
    )
    p.add_argument(
        "--format",
        choices=["md", "json", "both"],
        default="both",
        help="Which artifact(s) to write (default: both)",
    )
    p.add_argument(
        "--db",
        help="Path to the SQLite store (default: ARGUS_DB_PATH or data/starlink.db)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.week:
        year, week = parse_week_string(args.week)
    elif args.ending:
        d = datetime.fromisoformat(args.ending).replace(tzinfo=timezone.utc)
        iso = d.isocalendar()
        year, week = iso[0], iso[1]
    else:
        year, week = most_recent_complete_week()

    iso_week = f"{year}-W{week:02d}"
    start_ts, end_ts = iso_week_bounds(year, week)

    store = StarlinkStore(args.db) if args.db else StarlinkStore()
    report = build_report(store, start_ts, end_ts, iso_week=iso_week)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = load_previous_report(args.output_dir, iso_week)

    editor_notes: str | None = None
    if args.editor_notes:
        editor_notes = args.editor_notes.read_text()

    wrote: list[Path] = []
    if args.format in ("json", "both"):
        path = args.output_dir / f"{iso_week}.json"
        path.write_text(render_json(report))
        wrote.append(path)
    if args.format in ("md", "both"):
        path = args.output_dir / f"{iso_week}.md"
        path.write_text(render_markdown(report, previous=previous, editor_notes=editor_notes))
        wrote.append(path)

    print(f"[weekly] {iso_week}: {report['constellation']['total']:,} tracked, "
          f"{report['flagged_events']['total']} labels, "
          f"{report['data_quality']['fetch_attempts']} fetches")
    for p in wrote:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
