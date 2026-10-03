#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.7: record the v6 anchor's spread and the verdict it implies.

Applies the new measurement_stats API to the ONE comparison this project
currently has: the v8 re-run against the v6 anchor. That comparison is the
bead's whole justification -- it is how we know a single-run ratio resolves
nothing below the band.

Writes experiments/gpu-sim-u2o.6.6.7/metrics.json. Exits nonzero if the anchor
spread cannot be reconstructed or if the module's own invariants do not hold.

Every number here is DERIVED from the committed variance probe and the v6/v8
family files, then checked against each other. None is typed.
"""
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "metrics.json"
ANCHOR_DIR = REPO / "experiments" / "gpu-sim-u2o.6.6.2"

from epic_u2o.measurement_stats import (  # noqa: E402
    spread_from_runs, verdict_for_ratio,
)


def cells(path: Path) -> dict:
    return {r["label"]: r["median_ms"]
            for r in json.loads(path.read_text()).get("results", [])}


def main() -> int:
    probe_path = ANCHOR_DIR / "variance_probe.json"
    if not probe_path.exists():
        print(f"missing {probe_path}; merge 6.6.2 first", file=sys.stderr)
        return 1
    probe = json.loads(probe_path.read_text())

    # Rebuild the spread from the family files themselves rather than trusting
    # the probe's summary, then CHECK the two agree. A module that reads one
    # artifact and reports it back is a copy, not a check.
    runs = []
    for fam in ("memory", "reduce", "gemm"):
        runs.append((fam + ":v6", cells(ANCHOR_DIR / f"family_{fam}.json")))
        runs.append((fam + ":v8",
                     cells(ANCHOR_DIR / "variance_probe_v8" / f"family_{fam}.json")))

    # Per-family spread, because a pooled band hides that reduce moves ~10%
    # while gemm moves ~1%.
    per_family = {}
    for fam in ("memory", "reduce", "gemm"):
        a = cells(ANCHOR_DIR / f"family_{fam}.json")
        b = cells(ANCHOR_DIR / "variance_probe_v8" / f"family_{fam}.json")
        shared = sorted(set(a) & set(b))
        per_family[fam] = spread_from_runs(
            [(f"{fam}:v6", {k: a[k] for k in shared}),
             (f"{fam}:v8", {k: b[k] for k in shared})],
            baseline_id=f"v6-anchor:{fam}").to_dict()

    all_v6, all_v8 = {}, {}
    for fam in ("memory", "reduce", "gemm"):
        all_v6.update(cells(ANCHOR_DIR / f"family_{fam}.json"))
        all_v8.update(cells(ANCHOR_DIR / "variance_probe_v8" / f"family_{fam}.json"))
    shared = sorted(set(all_v6) & set(all_v8))
    overall = spread_from_runs(
        [("v6", {k: all_v6[k] for k in shared}),
         ("v8", {k: all_v8[k] for k in shared})],
        baseline_id="v6-anchor").to_dict()

    # The module's ratio is max/median across runs; the probe's is v8/v6 per
    # cell. They are different statistics, so this asserts DIRECTION and
    # MAGNITUDE agree rather than equality -- a genuine cross-check of two
    # independently written computations.
    probe_o = probe["overall_v8_over_v6"]
    agree = abs(overall["ratio_quantiles"]["median"] - probe_o["median"]) < 0.05

    # The practical consequence, computed rather than asserted in prose.
    examples = {f"ratio_{r}": verdict_for_ratio(r, spread_from_runs(
        [("v6", {k: all_v6[k] for k in shared}),
         ("v8", {k: all_v8[k] for k in shared})],
        "v6-anchor"))["verdict"] for r in (1.02, 1.04, 1.10, 1.25, 1.50)}

    metrics = {
        "bead": "gpu-sim-u2o.6.6.7",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "produced_by": "experiments/gpu-sim-u2o.6.6.7/measure_spread.py",
        "gpu_used": False,
        "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse",
                                  "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
        "purpose": ("A single-run ratio against a baseline resolves nothing "
                    "below that baseline's own run-to-run spread. This records "
                    "the spread beside the baseline so a comparison has to say "
                    "which side of the noise it falls on."),
        "overall_spread": overall,
        "per_family_spread": per_family,
        "cross_check_against_committed_probe": {
            "this_module_median": overall["ratio_quantiles"]["median"],
            "committed_probe_median": probe_o["median"],
            "agree_within_0.05": agree,
            "note": ("different statistics (max/median across runs here, v8/v6 "
                     "per cell there), so this asserts agreement in direction "
                     "and magnitude, not equality"),
        },
        "what_a_comparison_may_conclude": examples,
        "properties": {
            "anchor_band_is_wide": overall["ratio_quantiles"]["p90"] > 1.2,
            "four_percent_is_inconclusive":
                examples["ratio_1.04"] == "inconclusive",
            "fifty_percent_is_an_effect":
                examples["ratio_1.5"] == "effect_slower",
            "cross_check_agrees": agree,
            "no_single_run_spread": _single_run_refused(),
        },
    }
    metrics["all_properties_pass"] = all(metrics["properties"].values())
    OUT.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if not metrics["all_properties_pass"]:
        bad = [k for k, v in metrics["properties"].items() if not v]
        print(f"FAILED: {bad}", file=sys.stderr)
        return 1
    print("OK: the anchor's band is wide, a 4% single-run difference is "
          "inconclusive against it, a 50% one is not, and a one-run spread is "
          "refused rather than defaulted to zero.")
    return 0


def _single_run_refused() -> bool:
    """The module must REFUSE a one-run spread, not return zero.

    A zero-width band would make every later comparison look significant, which
    is the exact inversion this bead exists to stop.
    """
    try:
        spread_from_runs([("only", {"a": 1.0})], "x")
        return False
    except ValueError:
        return True


if __name__ == "__main__":
    sys.exit(main())