"""Event Investigator — deterministic analysis pipeline for flagged satellites.

This is Tier 1 of the SSA Agent system. Given a NORAD ID (typically from
the gap detector or a high-confidence rule_v1 label), it runs the same
multi-step investigation that was done manually for the Starlink-34343
debris event:

  1. Satellite metadata + TLE history
  2. Eccentricity / B* time series analysis (find jumps)
  3. Batch comparison (compare to launch group siblings)
  4. SatNOGS RF observation status
  5. Orbital neighborhood scan (new debris objects?)
  6. Severity assessment (critical / notable / routine)
  7. Draft Notable Flags paragraph (template-based, no LLM)

Design principle: deterministic first, LLM second. Every step produces
structured evidence. The draft paragraph is a template rendering of that
evidence. When an LLM layer is added later, it will improve the narrative
quality but will not change the underlying analysis.

Usage:
    python -m services.agent.investigator 64157
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from typing import Any

try:
    from services.telemetry.store import StarlinkStore
    from services.brain.orbital_analyzer import detect_new_neighbors
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from services.telemetry.store import StarlinkStore
    from services.brain.orbital_analyzer import detect_new_neighbors


# ── Analysis helpers ──

def _analyze_tle_timeseries(history: list[dict]) -> dict[str, Any]:
    """Find jumps and trends in the TLE time series."""
    if len(history) < 2:
        return {"status": "insufficient_data", "n_tles": len(history)}

    # history[0] is newest, history[-1] is oldest
    eccs = [(h.get("epoch_jd", 0), h.get("eccentricity") or 0) for h in history]
    bstars = [(h.get("epoch_jd", 0), h.get("bstar")) for h in history
              if h.get("bstar") is not None]
    mms = [(h.get("epoch_jd", 0), h.get("mean_motion") or 0) for h in history]

    # Biggest eccentricity jump between consecutive TLEs
    max_ecc_jump = 0.0
    jump_idx = None
    for i in range(len(eccs) - 1):
        # eccs[i] is newer, eccs[i+1] is older
        delta = abs(eccs[i][1] - eccs[i + 1][1])
        if delta > max_ecc_jump:
            max_ecc_jump = delta
            jump_idx = i

    result: dict[str, Any] = {
        "n_tles": len(history),
        "epoch_range_jd": [history[-1].get("epoch_jd", 0), history[0].get("epoch_jd", 0)],
        "ecc_range": [min(e[1] for e in eccs), max(e[1] for e in eccs)],
        "ecc_current": eccs[0][1],
        "mm_current": mms[0][1],
        "max_ecc_jump": max_ecc_jump,
    }

    if jump_idx is not None and max_ecc_jump > 0.001:
        result["ecc_jump_detected"] = True
        result["ecc_before"] = eccs[jump_idx + 1][1]  # older
        result["ecc_after"] = eccs[jump_idx][1]        # newer
        result["jump_epoch_jd"] = eccs[jump_idx][0]
    else:
        result["ecc_jump_detected"] = False

    if bstars:
        bstar_vals = [b[1] for b in bstars]
        result["bstar_range"] = [min(bstar_vals), max(bstar_vals)]
        result["bstar_current"] = bstars[0][1]
        # Sign changes (excluding near-zero noise)
        sign_changes = sum(
            1 for i in range(len(bstars) - 1)
            if (bstars[i][1] > 0) != (bstars[i + 1][1] > 0)
            and abs(bstars[i][1]) > 1e-5 and abs(bstars[i + 1][1]) > 1e-5
        )
        result["bstar_sign_changes"] = sign_changes

    return result


def _compare_to_batch(
    target_norad: int, target_history: list[dict], siblings: list[dict]
) -> dict[str, Any]:
    """Compare target satellite to its launch group siblings."""
    if not siblings or not target_history:
        return {"status": "no_data"}

    others = [s for s in siblings if s["norad_id"] != target_norad]
    if not others:
        return {"status": "no_siblings", "batch_size": 1}

    target = target_history[0]
    target_ecc = target.get("eccentricity") or 0
    target_bstar = target.get("bstar") or 0

    avg_ecc = sum(s.get("eccentricity") or 0 for s in others) / len(others)
    avg_mm = sum(s.get("mean_motion") or 0 for s in others) / len(others)
    avg_bstar = sum(s.get("bstar") or 0 for s in others) / len(others)

    return {
        "batch_size": len(others) + 1,
        "n_siblings": len(others),
        "batch_avg_ecc": avg_ecc,
        "target_ecc": target_ecc,
        "ecc_ratio": target_ecc / avg_ecc if avg_ecc > 1e-8 else None,
        "batch_avg_mm": avg_mm,
        "batch_avg_bstar": avg_bstar,
        "target_bstar": target_bstar,
        "bstar_ratio": abs(target_bstar) / abs(avg_bstar) if abs(avg_bstar) > 1e-8 else None,
    }


def _analyze_rf_status(observations: list[dict]) -> dict[str, Any]:
    """Summarize SatNOGS RF observation results."""
    if not observations:
        return {"status": "no_observations", "total_count": 0,
                "good_count": 0, "failed_count": 0, "unknown_count": 0}

    statuses = Counter(o.get("vetted_status", "unknown") for o in observations)
    return {
        "total_count": len(observations),
        "good_count": statuses.get("good", 0),
        "bad_count": statuses.get("bad", 0),
        "failed_count": statuses.get("failed", 0),
        "unknown_count": statuses.get("unknown", 0),
        "by_status": dict(statuses),
    }


def _assess_severity(evidence: dict) -> dict[str, Any]:
    """Score and classify the event severity."""
    score = 0
    reasons: list[str] = []

    tle = evidence.get("tle_analysis", {})
    if tle.get("ecc_jump_detected"):
        jump = tle.get("max_ecc_jump", 0)
        score += 3
        reasons.append(f"eccentricity jump {jump:.4f}")

    batch = evidence.get("batch_analysis", {})
    ecc_ratio = batch.get("ecc_ratio")
    ecc_jump = tle.get("ecc_jump_detected", False)
    if ecc_ratio and ecc_ratio > 5:
        if ecc_jump:
            # Sudden jump + high ratio = debris event signature
            score += 2
            reasons.append(f"eccentricity {ecc_ratio:.1f}× batch average (sudden jump)")
        else:
            # Gradual divergence = likely deorbiting, not debris
            score += 1
            reasons.append(f"eccentricity {ecc_ratio:.1f}× batch average (gradual — likely deorbit)")
    elif ecc_ratio and ecc_ratio > 2:
        score += 1
        reasons.append(f"eccentricity {ecc_ratio:.1f}× batch average")

    rf = evidence.get("rf_analysis", {})
    if rf.get("failed_count", 0) > 0 and rf.get("good_count", 0) == 0:
        score += 2
        reasons.append("RF observations failed, none good")
    elif rf.get("failed_count", 0) > rf.get("good_count", 0):
        score += 1
        reasons.append(f"RF: {rf['failed_count']} failed vs {rf['good_count']} good")

    neighbors = evidence.get("new_neighbors", [])
    if len(neighbors) > 3:
        score += 3
        reasons.append(f"{len(neighbors)} new objects in orbital neighborhood")
    elif len(neighbors) > 0:
        score += 1
        reasons.append(f"{len(neighbors)} new nearby object(s)")

    gap = evidence.get("gap_hours")
    if gap and gap > 72:
        score += 2
        reasons.append(f"TLE gap {gap:.0f}h")
    elif gap and gap > 24:
        score += 1
        reasons.append(f"TLE gap {gap:.0f}h")

    if score >= 5:
        severity = "critical"
    elif score >= 3:
        severity = "notable"
    else:
        severity = "routine"

    return {"severity": severity, "score": score, "reasons": reasons}


def _draft_paragraph(evidence: dict) -> str:
    """Generate a draft Notable Flags paragraph from evidence.

    Template-based, no LLM. When the LLM layer is added, it will replace
    this function but consume the same evidence dict.
    """
    sat = evidence.get("satellite") or {}
    name = sat.get("name") or f"NORAD {evidence['norad_id']}"
    norad = evidence["norad_id"]
    severity = evidence.get("severity", {})
    tle = evidence.get("tle_analysis", {})
    batch = evidence.get("batch_analysis", {})
    rf = evidence.get("rf_analysis", {})
    neighbors = evidence.get("new_neighbors", [])

    sentences: list[str] = []

    # Headline
    sev_label = severity.get("severity", "routine").upper()
    sentences.append(f"**{name}** ({norad}) — [{sev_label}]")

    # Eccentricity analysis
    ecc_ratio = batch.get("ecc_ratio")
    if tle.get("ecc_jump_detected"):
        sentences.append(
            f"Eccentricity jumped from {tle['ecc_before']:.5f} to "
            f"{tle['ecc_after']:.5f} (Δ={tle['max_ecc_jump']:.4f}) — "
            f"sudden change consistent with a debris event."
        )
    elif ecc_ratio and ecc_ratio > 5 and not tle.get("ecc_jump_detected"):
        sentences.append(
            f"Eccentricity is {ecc_ratio:.1f}× the batch average but "
            f"diverged gradually — consistent with deorbiting rather "
            f"than a debris event."
        )

    # Batch comparison
    ecc_ratio = batch.get("ecc_ratio")
    if ecc_ratio and ecc_ratio > 2:
        sentences.append(
            f"This is {ecc_ratio:.1f}× the average eccentricity of "
            f"{batch['n_siblings']} siblings from the same launch group."
        )

    # B* analysis
    if tle.get("bstar_current") is not None:
        bstar_ratio = batch.get("bstar_ratio")
        if bstar_ratio and bstar_ratio > 2:
            sentences.append(
                f"B* drag coefficient ({tle['bstar_current']:+.3e}) is "
                f"{bstar_ratio:.1f}× the batch average — possible loss of "
                f"attitude control or structural damage."
            )

    # RF status
    if rf.get("total_count", 0) > 0:
        sentences.append(
            f"SatNOGS ground stations: {rf['good_count']} good, "
            f"{rf['failed_count']} failed, {rf['unknown_count']} unvetted "
            f"observations."
        )
        if rf["failed_count"] > 0 and rf["good_count"] == 0:
            sentences.append(
                "No successful RF detections — consistent with satellite "
                "failure or tumbling debris."
            )

    # Neighbors
    if neighbors:
        sentences.append(
            f"{len(neighbors)} recently-cataloged object(s) detected in "
            f"the same orbital neighborhood — potential debris."
        )

    # Gap
    gap = evidence.get("gap_hours")
    if gap and gap > 24:
        sentences.append(f"TLE update gap: {gap:.0f} hours.")

    return "- " + " ".join(sentences)


# ── Main pipeline ──

def investigate_satellite(
    store: StarlinkStore,
    norad_id: int,
    context: str = "",
) -> dict[str, Any]:
    """Run the full investigation pipeline on a single satellite.

    Returns a structured evidence dict containing the analysis results
    from each step, a severity assessment, and a draft Notable Flags
    paragraph.
    """
    now = time.time()
    evidence: dict[str, Any] = {
        "norad_id": norad_id,
        "context": context,
        "investigated_at": now,
    }

    # Step 1: Satellite metadata
    sat = store.get_satellite(norad_id)
    evidence["satellite"] = sat

    if sat:
        last_seen = sat.get("last_seen") or now
        evidence["gap_hours"] = round((now - last_seen) / 3600, 1)

    # Step 2: TLE history analysis
    history = store.get_satellite_history(norad_id, limit=200)
    evidence["tle_analysis"] = _analyze_tle_timeseries(history)

    # Step 3: Batch comparison
    if sat and sat.get("intl_designator"):
        prefix = sat["intl_designator"][:5]
        siblings = store.get_batch_siblings(prefix)
        evidence["batch_analysis"] = _compare_to_batch(norad_id, history, siblings)
    else:
        evidence["batch_analysis"] = {"status": "no_designator"}

    # Step 4: SatNOGS RF status
    satnogs_obs = store.get_satnogs_observations(norad_id)
    evidence["rf_analysis"] = _analyze_rf_status(satnogs_obs)

    # Step 5: Orbital neighborhood scan
    neighbors = detect_new_neighbors(store, norad_id)
    evidence["new_neighbors"] = neighbors

    # Step 6: Severity assessment
    evidence["severity"] = _assess_severity(evidence)

    # Step 7: Draft paragraph
    evidence["draft"] = _draft_paragraph(evidence)

    elapsed_ms = int((time.time() - now) * 1000)
    evidence["elapsed_ms"] = elapsed_ms

    return evidence


def investigate_all_gaps(store: StarlinkStore) -> list[dict]:
    """Investigate all satellites currently flagged by the gap detector.

    Returns a list of evidence dicts, sorted by severity score (highest first).
    """
    from services.brain.orbital_analyzer import detect_tle_gaps
    gaps = detect_tle_gaps(store)
    results = []
    for gap in gaps:
        norad_id = gap["norad_id"]
        evidence = investigate_satellite(
            store, norad_id,
            context=f"TLE gap {gap.get('gap_hours', 0):.0f}h",
        )
        results.append(evidence)
    results.sort(key=lambda e: e.get("severity", {}).get("score", 0), reverse=True)
    return results


# ── CLI ──

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m services.agent.investigator",
        description="Investigate a satellite and produce an event assessment.",
    )
    parser.add_argument("norad_id", type=int, nargs="?",
                        help="NORAD catalog ID to investigate")
    parser.add_argument("--all-gaps", action="store_true",
                        help="Investigate all satellites with TLE gaps >24h")
    parser.add_argument("--db", help="Path to SQLite store")
    parser.add_argument("--json", action="store_true",
                        help="Output full evidence as JSON")
    args = parser.parse_args(argv)

    db_path = args.db or os.environ.get("ARGUS_DB_PATH", "data/starlink.db")
    store = StarlinkStore(db_path)

    if args.all_gaps:
        results = investigate_all_gaps(store)
        print(f"Investigated {len(results)} gapped satellites\n")
        for ev in results:
            sev = ev.get("severity", {})
            print(f"[{sev.get('severity', '?'):>8s}] score={sev.get('score', 0)}  "
                  f"{ev.get('satellite', {}).get('name') or ev['norad_id']}")
            print(f"  {ev.get('draft', '')}")
            print()
        return 0

    if not args.norad_id:
        parser.error("Provide a NORAD ID or use --all-gaps")

    evidence = investigate_satellite(store, args.norad_id)
    sev = evidence.get("severity", {})

    if args.json:
        import json
        # Remove non-serializable items
        clean = {k: v for k, v in evidence.items()
                 if k not in ("satellite",)}
        print(json.dumps(clean, indent=2, default=str))
    else:
        sat = evidence.get("satellite") or {}
        print(f"=== Investigation: {sat.get('name') or args.norad_id} "
              f"(NORAD {args.norad_id}) ===")
        print(f"Severity: {sev.get('severity', '?')} (score {sev.get('score', 0)})")
        print(f"Reasons: {', '.join(sev.get('reasons', []))}")
        print()

        tle = evidence.get("tle_analysis", {})
        print(f"TLE history: {tle.get('n_tles', 0)} records")
        if tle.get("ecc_jump_detected"):
            print(f"  ⚠ Eccentricity jump: {tle['ecc_before']:.6f} → {tle['ecc_after']:.6f} "
                  f"(Δ={tle['max_ecc_jump']:.4f})")
        print(f"  ecc range: {tle.get('ecc_range', [0,0])[0]:.6f} – {tle.get('ecc_range', [0,0])[1]:.6f}")
        if "bstar_current" in tle:
            print(f"  B* current: {tle['bstar_current']:+.3e}")

        batch = evidence.get("batch_analysis", {})
        if batch.get("ecc_ratio"):
            print(f"\nBatch comparison ({batch['batch_size']} siblings):")
            print(f"  ecc: {batch['target_ecc']:.6f} vs avg {batch['batch_avg_ecc']:.6f} "
                  f"({batch['ecc_ratio']:.1f}×)")

        rf = evidence.get("rf_analysis", {})
        if rf.get("total_count", 0) > 0:
            print(f"\nSatNOGS RF: {rf['good_count']} good / {rf['failed_count']} failed / "
                  f"{rf['unknown_count']} unknown")

        neighbors = evidence.get("new_neighbors", [])
        if neighbors:
            print(f"\nNew orbital neighbors: {len(neighbors)}")

        print(f"\nElapsed: {evidence.get('elapsed_ms', 0)}ms")
        print(f"\n--- Draft Notable Flag ---")
        print(evidence.get("draft", "(no draft)"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
