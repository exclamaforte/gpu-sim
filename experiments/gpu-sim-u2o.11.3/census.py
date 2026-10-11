#!/usr/bin/env python3
"""gpu-sim-u2o.11.3 regime census: counts first, verdict from a pre-registered rule.

The rule is stated BEFORE any stage 2-5 row is measured (it is committed with
this file), so the census cannot be tuned to fit the mix it finds:

  PASS iff  n_classes >= 3
        AND smallest-class share >= 5%
        AND memory->memory rows >= 100   (3x the current 32; a count, not a
            claim that coverage improved)

A total without this census fails acceptance item 9 by design: a row count
alone is how a launch-bound corpus gets mistaken for a training set (6.7.1
shipped 264/360 launch-bound cells, 73%, on its first pass).

Regime classes are 9.4's construction ({old-roofline bottleneck} ->
{new-roofline bottleneck}), recomputed live per row -- never a stored guess.
"""
import json
import sys
from pathlib import Path

MIN_CLASSES = 3
MIN_SMALLEST_SHARE = 0.05
MIN_MEMORY_BOUND = 100
MEMORY_CLASS = "memory->memory"


def census(counts: dict) -> dict:
    """Verdict over {regime_class: n_rows}. Pure; unit-tested with scratch mixes."""
    total = sum(counts.values())
    n_classes = len([c for c, n in counts.items() if n > 0])
    smallest_share = (min(counts.values()) / total) if total else 0.0
    mem = counts.get(MEMORY_CLASS, 0)
    reasons = []
    if n_classes < MIN_CLASSES:
        reasons.append(f"only {n_classes} regime classes (< {MIN_CLASSES})")
    if smallest_share < MIN_SMALLEST_SHARE:
        reasons.append(f"smallest-class share {smallest_share:.3f} "
                       f"(< {MIN_SMALLEST_SHARE})")
    if mem < MIN_MEMORY_BOUND:
        reasons.append(f"{MEMORY_CLASS} rows {mem} (< {MIN_MEMORY_BOUND})")
    return {"total": total, "per_class": dict(counts),
            "n_classes": n_classes, "smallest_share": round(smallest_share, 4),
            "verdict": "PASS" if not reasons else "FAIL",
            "reasons": reasons}


def require_census(artifact: dict) -> dict:
    """Acceptance item 9: a total alone FAILS. Returns the census block."""
    total = artifact.get("total_rows", artifact.get("n_measured"))
    cen = artifact.get("regime_census")
    if total is None:
        raise ValueError("no row count in artifact")
    if not isinstance(cen, dict) or not cen.get("per_class"):
        raise ValueError("row count without a regime census fails item 9: "
                         "a total alone is how a launch-bound corpus gets "
                         "mistaken for a training set")
    if sum(cen["per_class"].values()) != total:
        raise ValueError("census counts do not sum to the row total")
    return cen


def main() -> int:
    # Scratch negative control: a 95/3/2 mix must FAIL the rule.
    demo = {"latency->latency": 95, "latency->memory": 3, "memory->memory": 2}
    r = census(demo)
    print(json.dumps(r, indent=2))
    assert r["verdict"] == "FAIL", "census is green on a narrow mix: rule is decoration"
    assert any("share" in s for s in r["reasons"])
    # And a total-without-census artifact must fail item 9.
    try:
        require_census({"total_rows": 100})
    except ValueError as e:
        print(f"item-9 trap fires as designed: {e}")
    else:
        raise AssertionError("total-without-census passed item 9")
    print("NEGATIVE CONTROLS PASS: narrow mix flagged, total-alone rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
