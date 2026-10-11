"""gpu-sim-39u EXPERIMENT 2 (CPU-only): is the residual the n=262144 max floor?

zhq shipped LAUNCH_FLOOR_US = the max of 6.6.1's sub-2^18 floors. This
experiment replaces it with the N-LOCAL floor and re-runs the 456 rows.

Pre-registered rule (not fitted): the 6.6.1 floors are read AT RUNTIME from
the committed metrics.json (never re-typed); a row uses the floor recorded
at the largest n <= its problem_size, clamped at both ends:
  n <= 1024    -> 1.30832 | 1024 < n <= 16384  -> 1.3296 | n > 16384 -> 1.36528
(6.6.1 recorded only three sub-crossover points; the step is the honest
shape of three points, not an interpolation we did not measure.)

Decisive either way: 4% of floor cannot move a 73% error, so a null result
RULES OUT mechanism (b) for free; a large move CONFIRMS it. No GPU.

Also prints arm breakdowns for 3 sample rows (small 7.1 / mid / large e5q)
so the missing-constant-vs-units question is answered with numbers.
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))
import epic_u2o.roofline as roof  # noqa: E402

ZHQ = REPO / "experiments" / "gpu-sim-zhq" / "zhq_verify.py"
spec = importlib.util.spec_from_file_location("zhq_verify_mod", ZHQ)
zv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zv)

OLD = zv.load_old_roofline()

# Mechanism (b) input: IMPORTED from the committed measurement, not re-typed.
m661 = json.loads((REPO / "experiments/gpu-sim-u2o.6.6.1/metrics.json").read_text())
RTD = m661["noise_floor"]["reported_time_decomposition"]
FLOORS = sorted((int(n), RTD[n]["irreducible_launch_us"]) for n in RTD
                if int(n) <= 262144)
assert [n for n, _ in FLOORS] == [1024, 16384, 262144], FLOORS


def nlocal_floor_us(n):
    v = FLOORS[0][1]
    for nn, vv in FLOORS:
        if nn <= n:
            v = vv
    return v


def main():
    out = Path(sys.argv[sys.argv.index("--metrics-out") + 1]) \
        if "--metrics-out" in sys.argv else None
    rows = []
    for c in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl"):
        rows += [json.loads(l) for l in (REPO / c).read_text().splitlines()]

    err_ship, err_exp2 = [], []
    orig_latency = roof.calculate_latency_time
    demo_idx = {0, next(i for i, r in enumerate(rows) if r["problem_size"] >= 1 << 22),
                len(rows) - 1}
    for i, r in enumerate(rows):
        f = zv.row_features(r)
        actual = r["median_runtime_ms"]
        if actual <= 0:
            continue
        roof.calculate_latency_time = orig_latency
        p_ship = roof.roofline_predict(zv.to_new(f)).predicted_time_ms
        # Experiment 2: same model, n-local latency floor only.
        roof.calculate_latency_time = lambda _f, _n=r["problem_size"]: \
            nlocal_floor_us(_n) / 1000.0
        p_exp2 = roof.roofline_predict(zv.to_new(f)).predicted_time_ms
        err_ship.append(abs((p_ship - actual) / actual))
        err_exp2.append(abs((p_exp2 - actual) / actual))
        if i in demo_idx:
            fn2 = zv.to_new(f)
            occ2 = roof.calculate_occupancy(fn2)
            ct = roof.calculate_compute_time(fn2, occ2)
            mt = roof.calculate_memory_time(fn2, occ2)
            print(f"n={r['problem_size']} op={r['op_type']} "
                  f"compute={ct:.5f}ms mem={mt:.5f}ms "
                  f"lat={nlocal_floor_us(r['problem_size']) / 1000.0:.5f}ms "
                  f"pred={p_exp2:.5f}ms actual={actual:.5f}ms", flush=True)
    metrics = {"bead": "gpu-sim-39u", "n_rows": len(err_ship),
               "floors_imported": FLOORS,
               "mean_abs_shipped_floor": round(statistics.fmean(err_ship), 4),
               "mean_abs_nlocal_floor": round(statistics.fmean(err_exp2), 4),
               "verdict": "pass"}
    if out:
        out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"n={len(err_ship)} abs shipped {metrics['mean_abs_shipped_floor']} "
          f"-> n-local {metrics['mean_abs_nlocal_floor']} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
