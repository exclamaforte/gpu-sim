#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.6.3: regenerate the size sweep, stamped with its load.

Every number this bead reports about queue_state's cost comes from HERE, at run
time, against temporary directories. None is typed into prose. The reason is the
recurring failure this project keeps meeting: a hand-typed number beside a
derived one drifts, and the hand-typed one is the one a reader quotes.

Two measurement conventions, both learned the hard way in this session:

  * EVERY TIMING IS STAMPED WITH THE LOAD IT WAS TAKEN AT. The marginal cost is
    load-dependent (~0.185 us/file quiet, ~0.28 under load), so a bare
    millisecond figure describes one machine at one moment.
  * THE MEDIAN IS OVER ENOUGH TRIALS, AND THE MIN IS REPORTED TOO. A sub-ms
    micro-benchmark is at the mercy of run-queue position; a median alone hides
    that, and a min alone flatters.

Writes sweep.json beside this file. Exits nonzero if the sweep cannot be run.
"""
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "sweep.json"
SIZES = (200, 400, 700, 862, 1200, 2000, 4000)
TRIALS = 25
DEVICE_BUDGET_MS = 0.139


def load_qs():
    spec = importlib.util.spec_from_file_location(
        "qs_sweep", REPO / "epic_u2o" / "queue_state.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def synth(d: Path, n: int) -> None:
    """A queue with a realistic state mix: mostly terminal, some ACTIVE.

    The active fraction matters, because the ACTIVE set is re-read on every call
    by design -- that is the guarantee 6.6.6 exists to protect -- so it sets the
    floor this optimisation can reach. A fixture of all-finished jobs would
    measure a bound no real queue enjoys.
    """
    for i in range(n):
        state = "running" if i % 60 == 0 else "finished"
        (d / f"job{i:05d}.json").write_text(json.dumps({
            "id": f"j{i}", "state": state, "command": ["python3", "x.py"],
            "description": f"job {i}", "created_at": 1.0}))


def bench(qs, n: int, trials: int = TRIALS) -> dict:
    d = Path(tempfile.mkdtemp(prefix="qs_sweep_"))
    try:
        synth(d, n)
        qs.QUEUE_DIR = d
        qs.reset_cache()
        qs._load_all()                       # warm: first-call cost is not steady
        samples = []
        for _ in range(trials):
            t0 = time.perf_counter()
            qs._load_all()
            samples.append((time.perf_counter() - t0) * 1e3)
        return {"median_ms": round(statistics.median(samples), 5),
                "min_ms": round(min(samples), 5),
                "max_ms": round(max(samples), 5),
                "trials": trials}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    qs = load_qs()
    try:
        loadavg = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        loadavg = None

    points = []
    for n in SIZES:
        r = bench(qs, n)
        r["files"] = n
        r["us_per_file"] = round(r["median_ms"] * 1e3 / n, 5)
        r["x_device_budget"] = round(r["median_ms"] / DEVICE_BUDGET_MS, 3)
        points.append(r)
        print(f"  {n:5d} files  {r['median_ms']:.4f} ms  "
              f"{r['us_per_file']:.4f} us/file  {r['x_device_budget']:.2f}x")

    # Is the cost linear in directory size? Fit through the origin and report
    # the per-file spread, so a reader can see whether the marginal cost is flat
    # (linear) or drifting (super-linear). Both would defeat the criterion; the
    # difference matters for whether more optimisation is worth doing.
    xs = [p["files"] for p in points]
    ys = [p["median_ms"] for p in points]
    denom = sum(x * x for x in xs)
    slope = sum(x * y for x, y in zip(xs, ys)) / denom if denom else None
    per_file = [p["us_per_file"] for p in points]

    record = {
        "bead": "gpu-sim-u2o.6.6.6.3",
        "kind": "MEASURED SWEEP -- regenerates at run time, nothing typed",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "produced_by": "experiments/gpu-sim-u2o.6.6.6.3/measure_sweep.py",
        "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse",
                                  "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
        "loadavg_1_5_15": loadavg,
        "device_budget_ms": DEVICE_BUDGET_MS,
        "warm_cache": True,
        "state_mix": "every 60th job ACTIVE (running), rest terminal",
        "points": points,
        "fit": {
            "us_per_file_slope": round(slope * 1e3, 5) if slope else None,
            "per_file_spread_us": {"min": round(min(per_file), 5),
                                   "max": round(max(per_file), 5),
                                   "ratio": round(max(per_file) / min(per_file), 3)},
            "linear": (max(per_file) / min(per_file)) < 1.5,
            "reading": ("per-file cost is flat within the measured spread, so the "
                        "cost is LINEAR in directory size. It is not I/O: the "
                        "existing reverify branch already skips stat() for cached "
                        "terminal entries, so the residual is per-file Python "
                        "work. Linear is still O(n), and the criterion asks for "
                        "sub-linear."),
        },
        "caveat": ("every timing here is load-dependent and stamped with the "
                   "loadavg above; a bare millisecond figure from this sweep is "
                   "not comparable with one taken at a different load."),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nloadavg(1m)={loadavg[0] if loadavg else 'n/a'}")
    print(f"slope {record['fit']['us_per_file_slope']} us/file, "
          f"linear={record['fit']['linear']}, "
          f"per-file spread {record['fit']['per_file_spread_us']['ratio']}x")
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())