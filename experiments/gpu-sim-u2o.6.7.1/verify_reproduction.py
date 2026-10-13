#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1: run acceptance item 2's reproduction check ON DEMAND.

WHY THIS EXISTS. The reproduction check is marked `expectedFailure` in
tests/test_batched_worker.py so that merging this branch does not put a designed
failure into the repository's default signal. That is the right call for the
default suite and the wrong call if nobody can ask the question on demand -- a
decorator is not a check anyone runs.

So this runs the same comparison and EXITS NONZERO while item 2 is unmet, and
prints the measured numbers. It is the opposite of the default suite: quiet by
default, loud when asked.

An expectedFailure is not a skip: if the reproduction ever starts passing, the
suite reports a FAILURE. That is the property that makes the default signal
trustworthy -- the debt cannot be forgotten by going green.

Usage: python3 experiments/gpu-sim-u2o.6.7.1/verify_reproduction.py
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOL_FLOOR, TOL_CAP, TOL_SIGMA = 0.10, 0.75, 3.0


def _rows():
    p = HERE / "batch_rows.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] \
        if p.exists() else []


REFERENCE_REF = ("lab/gpu-sim-u2o.7.1:"
                 "experiments/gpu-sim-u2o.7.1/measured.jsonl")


def _single():
    """7.1's committed medians, read from the same ref the suite reads.

    Done with `git show` directly rather than by importing the test module in a
    subprocess: my first version shelled out and silently returned {} when the
    subprocess raised, which is the worst failure shape for a script whose whole
    job is to report a number -- it looked like 'no data' rather than 'I broke'.
    """
    r = subprocess.run(["git", "show", REFERENCE_REF], cwd=str(REPO),
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"cannot read {REFERENCE_REF}: {r.stderr.strip()[:160]}",
              file=sys.stderr)
        return {}
    out = {}
    for line in r.stdout.splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["kernel_hash"]] = d
    return out


def main() -> int:
    rows, single = _rows(), _single()
    if not rows or not single:
        print(f"cannot run: {len(rows)} rows, {len(single)} reference medians",
              file=sys.stderr)
        return 2
    rels, reproducing, worst = [], 0, None
    for r in rows:
        s = single.get(r["kernel_hash"])
        if not s:
            continue
        b = s["median_runtime_ms"]
        rel = abs(r["median_runtime_ms"] - b) / b
        tol = max(TOL_FLOOR, min(TOL_SIGMA * (s.get("ai_cv") or 0.0), TOL_CAP))
        rels.append(rel)
        if rel <= tol:
            reproducing += 1
        elif worst is None or rel > worst[0]:
            worst = (rel, r["kernel_hash"], tol)
    frac = reproducing / len(rels) if rels else 0.0
    verdict = "MET" if frac >= 0.90 else "OPEN"
    print(f"acceptance item 2 (median reproduction): {verdict}")
    print(f"  {reproducing}/{len(rels)} medians reproduce 7.1's ({frac:.3f}, need 0.90)")
    if worst:
        print(f"  worst: {worst[0]*100:.0f}% over a {worst[2]*100:.0f}% "
              f"tolerance ({worst[1]})")
    print(f"  measured {datetime.now(timezone.utc).isoformat()}")
    if verdict != "MET":
        print("\nBLOCKER 2 IS OPEN. The batched worker's medians do not reproduce")
        print("the single-run reference within the measured noise floor. Cause not")
        print("yet found. This is why the bead cannot be validated.")
    return 0 if verdict == "MET" else 1


if __name__ == "__main__":
    sys.exit(main())
