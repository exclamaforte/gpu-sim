#!/usr/bin/env python3
"""gpu-sim-7o2: median/IQR per cell from interleaved reps; answers the anomaly.

Reads one or more reps.jsonl files, prints per-cell n/median/IQR of step_ms,
and tests d=512 TP2+DP2 vs d=1024 TP2+DP2. Writes nothing tracked.
"""
import json
import statistics
import sys


def iqr(xs):
    qs = statistics.quantiles(xs, n=4, method="exclusive")
    return qs[2] - qs[0], qs[0], qs[2]


def main(paths):
    cells = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r.get("rc") == 0 and r.get("step_ms") is not None:
                    cells.setdefault(r["cell"], []).append(r["step_ms"])
    for name in sorted(cells):
        xs = sorted(cells[name])
        med = statistics.median(xs)
        spread, q1, q3 = iqr(xs) if len(xs) >= 4 else (float("nan"), float("nan"), float("nan"))
        print(f"{name:12s} n={len(xs):2d} median={med:7.1f} IQR={spread:6.1f} "
              f"[{q1:.1f},{q3:.1f}] min={xs[0]:.1f} max={xs[-1]:.1f}")
    a = sorted(cells.get("d512-TP2DP2", []))
    b = sorted(cells.get("d1024-TP2DP2", []))
    if a and b:
        ma, mb = statistics.median(a), statistics.median(b)
        print(f"\nanomaly: d512-TP2 median {ma:.1f} vs d1024-TP2 median {mb:.1f} "
              f"-> {'REPRODUCES (real effect)' if ma > mb else 'NOT reproduced (contention)'}")
        # Non-overlap of IQR boxes = the difference survives contention spread.
        _, a1, a3 = iqr(a)
        _, b1, b3 = iqr(b)
        print(f"IQR boxes: d512 [{a1:.1f},{a3:.1f}] d1024 [{b1:.1f},{b3:.1f}] "
              f"-> {'disjoint' if (a1 > b3 or b1 > a3) else 'overlapping'}")


if __name__ == "__main__":
    main(sys.argv[1:])
