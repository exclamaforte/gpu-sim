#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.2: CROSS-RUN VARIANCE PROBE. This is NOT an anchor.

WHAT THIS IS. The anchor is v6 (four families, one run, job 032717-*), and the
PI ruled it stays authoritative. The v8 re-run re-measured three of the four
families on a later run. Comparing them tells us how much the SAME
configuration moves between two runs -- which is the number every later
single-run comparison against this anchor needs in order to mean anything.

WHY IT IS EXPLICITLY NOT AN ANCHOR. v8 is missing its launch family (see
gpu-sim-u2o.6.6.2.1), so it cannot be one: an anchor is a coherent set measured
in one run, and v8 is three families from run 053729-* beside a launch family
from run 032717. Promoting it would be the mixed-provenance artifact this
project was reopened over. It is a sibling measurement, used here only to
estimate run-to-run reproducibility.

THE POINT OF THE WHOLE EXERCISE. A comparison against the anchor that does not
state this spread is meaningless: a 5% median difference between two runs of
the same code is indistinguishable from a real effect. So the spread is
recorded as a first-class artifact rather than discovered later.

Reads:  the committed v6 family files (in this tree)
        the v8 family files preserved under variance_probe_v8/
Writes: variance_probe.json next to this script. Exits nonzero if the probe
        cannot be computed.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V6_DIR = HERE                      # the committed anchor family files
V8_DIR = HERE / "variance_probe_v8"  # the preserved later run

FAMILIES = ("memory", "reduce", "gemm")   # launch is absent from v8; see 6.6.2.1


def load(path: Path):
    return json.loads(path.read_text())


def cells(doc: dict) -> dict:
    """label -> median_ms, the comparable key. Label is stable across runs
    because it is the config string the measurement emits, so it is the join."""
    return {r["label"]: r["median_ms"] for r in doc.get("results", [])}


def quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    def q(p):
        if len(xs) == 1:
            return xs[0]
        k = (len(xs) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return {"min": round(qs(xs)[0] if False else xs[0], 4),
            "p10": round(q(0.10), 4),
            "median": round(q(0.50), 4),
            "p90": round(q(0.90), 4),
            "max": round(xs[-1], 4)}


def main() -> int:
    per_family = {}
    all_ratios = []
    missing = []

    for fam in FAMILIES:
        p6, p8 = V6_DIR / f"family_{fam}.json", V8_DIR / f"family_{fam}.json"
        if not p8.exists():
            missing.append(str(p8))
            continue
        d6, d8 = load(p6), load(p8)
        c6, c8 = cells(d6), cells(d8)
        shared = sorted(set(c6) & set(c8))
        ratios = [c8[k] / c6[k] for k in shared if c6[k]]
        all_ratios += ratios
        per_family[fam] = {
            "cells_v6": len(c6),
            "cells_v8": len(c8),
            "comparable_cells": len(shared),
            "v8_over_v6": quantiles(ratios),
            "v6_run_job": (d6.get("queue_state_before") or {}).get("own_job_id"),
            "v8_run_job": (d8.get("queue_state_before") or {}).get("own_job_id"),
        }

    if not all_ratios:
        print("no comparable cells; cannot compute the probe", file=sys.stderr)
        return 1

    overall = quantiles(all_ratios)
    record = {
        "bead": "gpu-sim-u2o.6.6.2",
        "kind": "VARIANCE PROBE -- NOT AN ANCHOR",
        "not_an_anchor_because": (
            "v8 lacks its launch family (gpu-sim-u2o.6.6.2.1), so it is not a "
            "coherent single-run set. The anchor remains v6. This file exists "
            "only to measure how far the same configuration moves between two "
            "runs, so that a later single-run comparison against the anchor "
            "can state its resolution."),
        "question": "how far does the SAME configuration move between two runs?",
        "anchor_run": "v6, 20260928t032717-* (four families, one run)",
        "probe_run": "v8, 20260928t053729-*/053730-* (three families)",
        "join_key": "the config label emitted by measure_family.py",
        "per_family": per_family,
        "overall_v8_over_v6": overall,
        "median_drift_pct": round(100 * (overall["median"] - 1), 2),
        "interpretation": (
            f"median {overall['median']}x with p90 {overall['p90']}x and max "
            f"{overall['max']}x across {len(all_ratios)} comparable cells. A "
            f"single-run comparison against the v6 anchor therefore cannot "
            f"resolve anything smaller than roughly the p90 band without "
            f"replicates. This is the number 6.6.6.1's criterion-1 ratios and "
            f"any future A/B must quote."),
        "missing_inputs": missing,
        "UNRECONCILED_SECOND_COMPUTATION": {
            "why_this_is_here": (
                "The PI independently differenced v8 against v6 and reported 30 "
                "comparable cells, overall median 1.047 (p10 1.00, p90 1.355), "
                "memory median 1.082 (max 1.355). This script's join -- the "
                "config label emitted by measure_family.py -- yields 106 cells "
                "and memory median 1.0177. The two AGREE EXACTLY on reduce "
                "(1.1012 vs 1.101) and gemm (1.0133 vs 1.013), and DISAGREE on "
                "the memory family and on the cell count. Both cannot be "
                "right and I could not reproduce 30 from this data: memory is "
                "90 cells (6 kernel kinds x 15 configs), reduce 8, gemm 8."),
            "this_script_figures": "106 cells, memory median 1.0177, overall median 1.0181",
            "verified_by": (
                "hand-check of four cells against the raw files "
                "(e.g. 'copy n=1024 1block/1024' v6=0.002768 v8=0.003104 "
                "ratio 1.1214), and a confirmation that all 90 memory labels "
                "appear in both files with zero one-sided labels"),
            "status": "UNRESOLVED. The join key is stated above so a reader can "
                      "reproduce this number exactly; the PI's 30-cell selection "
                      "is not reproducible from this data and needs naming "
                      "before either figure is quoted as the anchor's "
                      "resolution.",
            "does_it_change_the_ruling": (
                "No. Both computations say the same thing qualitatively: a few "
                "percent median drift with a tail past 1.5x. So (c) stands and "
                "a single-run comparison against v6 still cannot resolve "
                "anything below the p90 band without replicates. Only the "
                "number to quote is in question."),
                },
    }
    (HERE / "variance_probe.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())