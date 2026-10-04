#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1 ABAB: does the harness path move device-side medians?

Contrast (PI-approved 2026-10-04): two cells x two arms, interleaved ABAB,
>=3 invocations per arm per cell, 20 trials each, same binaries, same
estimator (median of per-invocation medians):
  A: full run_kernel() path (wrapper + inner + preexec + cgroup + scratch)
  B: same binary via bare `timeout 60` subprocess (no harness at all)

Cells: worst-rel corner hash 71cbc01535bc8a11 (rel 0.45) + the smallest
launch cell 9c9e5c9c245bf8a6 (scale, n=4096).

Falsification (stated before the run): the harness hypothesis predicts
|A-B|/B > 5% SAME-SIGN on BOTH cells. Split cells, <5%, or opposite signs
count as EXONERATION, not as 'needs more runs'. Interleaving cancels drift;
no post-hoc arm selection.

One ts dose, --max-runtime 10m cap. Per-invocation queue_state + telemetry
incl. trial counts (the corner gap, closed here).
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from epic_u2o.safe_harness import SafeHarness  # noqa: E402

BUILD_DIR = HERE / "corner_build"
CELLS = ["71cbc01535bc8a11", "9c9e5c9c245bf8a6"]
ROUNDS = 3
BARE_TIMEOUT = 60


def trials_from(stdout: str) -> list:
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "time_ms" in line:
            try:
                out.append(float(json.loads(line)["time_ms"]))
            except (ValueError, KeyError):
                pass
    return out


def telemetry():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return {"rc": r.returncode, "csv": r.stdout.strip()}
    except Exception as e:
        return {"error": str(e)}


def queue_sample(tag: str) -> dict:
    from epic_u2o import queue_state as qs
    q = qs.queue_state(own_description_substr="u2o.6.7.1",
                       own_command_substr="ab_discrimination.py")
    return {"tag": tag,
            "own_job_id": q.get("own_job_id"),
            "n_other_active": q.get("n_other_active"),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def run_A(h: SafeHarness, exe: str) -> tuple:
    """Full harness path. Returns (trials, note)."""
    r = h.run_kernel(exe)
    if not r.success:
        return [], f"harness success=False rc={r.returncode}"
    return trials_from(r.stdout), ""


def run_B(exe: str) -> tuple:
    """Bare path: same binary, `timeout` backstop, no harness at all."""
    try:
        r = subprocess.run(["timeout", str(BARE_TIMEOUT), exe],
                           capture_output=True, text=True, timeout=BARE_TIMEOUT + 30)
    except subprocess.TimeoutExpired:
        return [], "bare path exceeded timeout 60 (ANOMALY)"
    if r.returncode != 0:
        return [], f"bare rc={r.returncode} stderr={r.stderr[:200]}"
    return trials_from(r.stdout), ""


def main() -> int:
    if "--list-cells" in sys.argv:
        for c in CELLS:
            p = BUILD_DIR / f"k_{c}.out"
            print(c, "binary_present=" + str(p.exists()))
        return 0
    for c in CELLS:
        if not (BUILD_DIR / f"k_{c}.out").exists():
            print(f"FATAL: missing binary for {c}; run corner Phase-A build.",
                  flush=True)
            return 2
    h = SafeHarness(timeout_sec=300)
    q0 = queue_sample("start")
    print(f"ABAB: {len(CELLS)} cells x {ROUNDS} rounds x 2 arms")
    print(f"queue at start: own={q0['own_job_id']} "
          f"n_other={q0['n_other_active']}")
    if (q0["n_other_active"] or 0) > 0:
        print("ABORT: n_other>0 at start -- no dose on a contended baseline.",
              flush=True)
        return 2
    per_cell = {}
    t_start = time.time()
    with (HERE / "ab_rows.jsonl").open("w") as fh:
        for cell in CELLS:
            exe = str(BUILD_DIR / f"k_{cell}.out")
            arms = {"A": [], "B": []}
            for rnd in range(ROUNDS):
                for arm, fn in (("A", lambda: run_A(h, exe)),
                                ("B", lambda: run_B(exe))):
                    t0 = time.time()
                    trials, note = fn()
                    row = {"cell": cell, "arm": arm, "round": rnd,
                           "n_trials": len(trials),
                           "median_ms": statistics.median(trials) if trials else None,
                           "note": note,
                           "wall_s": round(time.time() - t0, 2),
                           "queue_state": queue_sample(f"{cell}-{arm}{rnd}"),
                           "telemetry": telemetry()}
                    fh.write(json.dumps(row) + "\n")
                    if trials:
                        arms[arm].append(statistics.median(trials))
                    print(f"  {cell[:8]} {arm}{rnd}: "
                          f"{'%.4f' % statistics.median(trials) if trials else 'NO-DATA'} "
                          f"ms n={len(trials)} {note}", flush=True)
            per_cell[cell] = arms
    cells = {}
    for cell, arms in per_cell.items():
        if len(arms["A"]) < 2 or len(arms["B"]) < 2:
            cells[cell] = {"error": "missing arm data",
                           "n_A": len(arms["A"]), "n_B": len(arms["B"])}
            continue
        mA = statistics.median(arms["A"])
        mB = statistics.median(arms["B"])
        ratio = mA / mB if mB else None
        cells[cell] = {"median_A_ms": mA, "median_B_ms": mB,
                       "ratio_A_over_B": ratio,
                       "A_medians": arms["A"], "B_medians": arms["B"]}
    ratios = [v["ratio_A_over_B"] for v in cells.values()
              if v.get("ratio_A_over_B")]
    implicated = (len(ratios) == len(CELLS) and
                  all(abs(r - 1.0) > 0.05 for r in ratios) and
                  (all(r > 1.0 for r in ratios)
                   or all(r < 1.0 for r in ratios))) if ratios else False
    verdict = {
        "criterion": "|A-B|/B > 5% same-sign on BOTH cells implicates "
                     "the harness path; else EXONERATED",
        "same_sign_over_5pct": implicated,
        "reading": ("harness path IMPLICATED" if implicated
                    else "harness path EXONERATED -- next: binary forensics"),
    }
    summary = {"bead": "gpu-sim-u2o.6.7.1", "arm": "ABAB harness-vs-bare",
               "cells": cells, "verdict": verdict,
               "wall_s": round(time.time() - t_start, 1),
               "tree_rev": subprocess.run(
                   ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                   capture_output=True, text=True).stdout.strip()}
    (HERE / "ab_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
