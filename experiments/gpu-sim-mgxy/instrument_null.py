#!/usr/bin/env python3
"""gpu-sim-mgxy: instrument the cell that produced kernel_us < 0.

The bead asks which hypothesis makes a kernel measure FASTER than the empty
kernel. The decisive, cheap experiment is the NULL cell: run the SAME empty
binary twice, treat the two series as (scale, empty) and compute the estimator.
A null cell cannot be faster than itself, so anything non-zero there is the
estimator's own noise, and the magnitude/sign-rate says whether a small real
kernel can go negative by construction.

It runs, per repeat:
  E  = empty binary, 20 trials      (floor, as the current estimator sees it)
  E2 = the same empty binary again  (null "scale" series)
  S  = a real tiny kernel binary    (the cell under study)
and reports kernel_us for (E2,E) and (S,E) under
  * median  : med(scale) - med(empty)            <- current measure.py
  * warm_min: med(warm(scale)) - min(warm(empty)) <- candidate
where warm() discards the first WARM launches.

Usage: python3 experiments/gpu-sim-mgxy/instrument_null.py [--repeats N]
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import os
BIN = Path(os.environ.get("GPU_SIM_8ZQR_BIN",
                         ROOT / "experiments" / "gpu-sim-8zqr" / "build" / "bin"))
EMPTY = BIN / os.environ.get("GPU_SIM_EMPTY_BIN", "empty_16_1_1_64_1_1.out")
SCALE = BIN / os.environ.get("GPU_SIM_SCALE_BIN", "k_divergent_branch_0000.out")
WARM = 5


def run_trials(exe):
    """Launch the binary and collect its per-trial time_ms in order."""
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    out = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if "time_ms" in o:
            out.append(o["time_ms"])
    if r.returncode != 0 or not out:
        raise RuntimeError(f"{exe.name} produced no trials: rc={r.returncode} {r.stderr[-300:]}")
    return out


def med(xs):
    return statistics.median(xs) if xs else 0.0


def warm(xs):
    return xs[WARM:] if len(xs) > WARM * 2 else xs


def est_median(scale, empty):
    return (med(scale) - med(empty)) * 1000.0


def est_warm_min(scale, empty):
    w = warm(empty)
    return (med(warm(scale)) - min(w)) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    null_med, null_wm, real_med, real_wm = [], [], [], []
    for i in range(args.repeats):
        e = run_trials(EMPTY)
        e2 = run_trials(EMPTY)
        s = run_trials(SCALE)
        nm, nw = est_median(e2, e), est_warm_min(e2, e)
        rm, rw = est_median(s, e), est_warm_min(s, e)
        null_med.append(nm); null_wm.append(nw)
        real_med.append(rm); real_wm.append(rw)
        print(json.dumps({"repeat": i, "null_median_us": round(nm, 3),
                          "null_warm_min_us": round(nw, 3),
                          "real_median_us": round(rm, 3),
                          "real_warm_min_us": round(rw, 3),
                          "empty_trials": [round(x * 1000, 3) for x in e],
                          "scale_trials": [round(x * 1000, 3) for x in s]}), flush=True)

    def summary(name, xs, limit_us=0.0):
        neg = sum(1 for x in xs if x < 0)
        print(f"{name}: n={len(xs)} median={med(xs):.3f}us "
              f"min={min(xs):.3f} max={max(xs):.3f} "
              f"negative={neg}/{len(xs)} ({100.0 * neg / len(xs):.0f}%) "
              f"below_{limit_us}us={sum(1 for x in xs if x < limit_us)}")

    print("\n=== summary ===")
    summary("NULL cell, median estimator   ", null_med)
    summary("NULL cell, warm-min estimator ", null_wm)
    summary("REAL cell, median estimator   ", real_med)
    summary("REAL cell, warm-min estimator ", real_wm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
