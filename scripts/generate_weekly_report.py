#!/usr/bin/env python3
"""One-command weekly report generation — no manual pre-work.

Runs the full pipeline:
  1. Resolve expired predictions → update scorecard
  2. Generate new predictions from current signals
  3. Run investigator on top anomalies + gap satellites
  4. Auto-draft editor notes from investigator output
  5. Generate the report (MD + JSON)

Usage (on VPS):
  docker compose exec -T api python scripts/generate_weekly_report.py

Usage (locally with a DB snapshot):
  python scripts/generate_weekly_report.py --db /tmp/argus.db

The output is a complete report ready for human review + Substack.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make imports work from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.telemetry.store import StarlinkStore
from services.report.weekly import (
    build_report, render_markdown, render_substack_markdown, render_json,
    most_recent_complete_week, iso_week_bounds, load_previous_report,
)
from services.report.predictions import generate_predictions, resolve_predictions
from services.agent.investigator import investigate_all_gaps, investigate_satellite
from services.brain.orbital_analyzer import detect_tle_gaps


def _auto_editor_notes(store: StarlinkStore) -> str:
    """Generate editor notes automatically from investigator + predictions.

    Produces the Notable Flags section without any human input. The human
    can review and edit before publishing, but the default output is
    publication-ready.
    """
    sections: list[str] = []

    # 1. Investigate gap satellites (most newsworthy)
    print("  [auto-notes] investigating gap satellites...")
    gap_results = investigate_all_gaps(store)
    critical = [r for r in gap_results if r.get("severity", {}).get("severity") == "critical"]
    notable = [r for r in gap_results if r.get("severity", {}).get("severity") == "notable"]

    if critical:
        for ev in critical[:3]:
            sections.append(ev.get("draft", ""))

    if notable and len(sections) < 3:
        for ev in notable[:2]:
            sections.append(ev.get("draft", ""))

    # 2. Top anomalies by confidence (if we have room)
    if len(sections) < 3:
        print("  [auto-notes] scanning top anomalies...")
        anomalies = store.get_anomalies(limit=100)
        # Filter to recent + high confidence, exclude what we already covered
        covered_norads = {r["norad_id"] for r in gap_results}
        top_anoms = [
            a for a in anomalies
            if a.get("confidence", 0) > 0.8
            and a["norad_id"] not in covered_norads
            and a.get("classified_by") == "rule_v1"
        ][:5]

        for a in top_anoms:
            if len(sections) >= 4:
                break
            name = a.get("name") or f"NORAD {a['norad_id']}"
            conf = a.get("confidence", 0)
            details = a.get("details", "")
            cause = a.get("cause", "")
            sections.append(
                f"- **{name}** ({a['norad_id']}) — `{cause}` "
                f"(conf {conf:.2f}). {details}"
            )

    # 3. Prediction scorecard (if any resolved)
    scorecard = store.get_prediction_scorecard()
    if scorecard.get("correct", 0) + scorecard.get("incorrect", 0) > 0:
        acc = scorecard.get("accuracy")
        acc_str = f"{acc*100:.0f}%" if acc is not None else "N/A"
        sections.append(
            f"- **Prediction scorecard**: {scorecard['correct']} correct, "
            f"{scorecard['incorrect']} incorrect out of "
            f"{scorecard['correct'] + scorecard['incorrect']} resolved "
            f"({acc_str} accuracy). {scorecard['pending']} predictions "
            f"still pending."
        )

    if not sections:
        sections.append(
            "- _No critical or notable events detected this week. "
            "All monitored satellites operating within normal parameters._"
        )

    return "\n\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the weekly report — fully automated, one command.",
    )
    parser.add_argument("--week", help="ISO week (e.g. 2026-W16). Default: most recent complete week.")
    parser.add_argument("--db", help="Path to SQLite store")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"),
                        help="Output directory (default: reports/)")
    parser.add_argument("--skip-predictions", action="store_true",
                        help="Skip prediction resolve/generate step")
    parser.add_argument("--editor-notes", type=Path,
                        help="Override auto-generated editor notes with this file")
    args = parser.parse_args(argv)

    db_path = args.db or os.environ.get("ARGUS_DB_PATH", "data/starlink.db")
    store = StarlinkStore(db_path)

    # Determine week. Default: the CURRENT ISO week (not the previous
    # complete one), because this script is designed to run via cron on
    # Sunday evening when the current week is 96% complete. The
    # min(end_ts, now) clamp in build_report handles the remaining hours.
    if args.week:
        year_str, week_str = args.week.split("-W")
        year, week = int(year_str), int(week_str)
    else:
        now_utc = datetime.now(timezone.utc)
        iso = now_utc.isocalendar()
        year, week = iso[0], iso[1]
    iso_week = f"{year}-W{week:02d}"
    start_ts, end_ts = iso_week_bounds(year, week)

    print(f"[weekly] Generating {iso_week}")
    print(f"  window: {datetime.fromtimestamp(start_ts, tz=timezone.utc).date()} → "
          f"{datetime.fromtimestamp(end_ts, tz=timezone.utc).date()}")

    # Step 1: Resolve expired predictions
    if not args.skip_predictions:
        print("\n[step 1/5] Resolving expired predictions...")
        resolved = resolve_predictions(store)
        print(f"  resolved {len(resolved)} predictions")
        for r in resolved[:5]:
            print(f"    [{r['outcome']:>9s}] {r.get('description', '')[:60]}")

    # Step 2: Generate new predictions
    if not args.skip_predictions:
        print("\n[step 2/5] Generating new predictions...")
        new_preds = generate_predictions(store)
        print(f"  generated {len(new_preds)} new predictions")

    # Step 3: Auto-generate editor notes (or use provided file)
    print("\n[step 3/5] Preparing editor notes...")
    if args.editor_notes and args.editor_notes.exists():
        editor_notes = args.editor_notes.read_text()
        print(f"  using provided notes from {args.editor_notes}")
    else:
        editor_notes = _auto_editor_notes(store)
        print(f"  auto-generated {len(editor_notes)} chars")

    # Step 4: Build report
    print("\n[step 4/5] Building report...")
    report = build_report(store, start_ts, end_ts, iso_week=iso_week)
    print(f"  {report['constellation']['total']:,} tracked, "
          f"{report['flagged_events']['total']} labels, "
          f"{report['data_quality']['fetch_attempts']} fetches")

    # Load previous week for deltas
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = load_previous_report(args.output_dir, iso_week)
    if previous:
        print(f"  loaded previous week for delta comparison")

    # Step 5: Render and save
    print("\n[step 5/5] Rendering...")
    json_path = args.output_dir / f"{iso_week}.json"
    md_path = args.output_dir / f"{iso_week}.md"
    substack_path = args.output_dir / f"{iso_week}-substack.md"
    notes_path = args.output_dir / f"{iso_week}-editor-notes.md"

    json_path.write_text(render_json(report))
    md_path.write_text(render_markdown(report, previous=previous, editor_notes=editor_notes))
    substack_path.write_text(
        render_substack_markdown(report, previous=previous, editor_notes=editor_notes)
    )
    notes_path.write_text(editor_notes)

    print(f"\n  wrote {json_path}")
    print(f"  wrote {md_path}")
    print(f"  wrote {substack_path}")
    print(f"  wrote {notes_path}")

    # Summary
    sc = store.get_prediction_scorecard()
    print(f"\n{'='*50}")
    print(f"  {iso_week} report ready for review")
    print(f"  {report['constellation']['total']:,} satellites tracked")
    print(f"  {report['flagged_events']['total']} anomaly labels")
    print(f"  {sc['total']} predictions ({sc['pending']} pending)")
    if sc.get("accuracy") is not None:
        print(f"  prediction accuracy: {sc['accuracy']*100:.0f}%")
    print(f"{'='*50}")
    print(f"\nNext steps:")
    print(f"  1. Review {md_path}")
    print(f"  2. Edit editor notes if needed")
    print(f"  3. Post to argusorb.substack.com")

    return 0


if __name__ == "__main__":
    sys.exit(main())
