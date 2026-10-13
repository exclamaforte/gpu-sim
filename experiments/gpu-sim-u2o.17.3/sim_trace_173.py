#!/usr/bin/env python3
"""gpu-sim-u2o.17.3: simulated-trace path driven by REAL per-op-class
predictions (CPU-only, no GPU -- consumes 17.5's committed fixture).

Method: per op class, leave-one-size-out log-log fit of device_ms_median
vs case size (elements; matmul uses flops=2n^3), predict the held-out
size, report relative error. Op classes with <3 sizes cannot fit (gap,
recorded). Naive roofline (marketing constants, same as the gate script)
runs on the same held-out cases for a with-numbers comparison.
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "experiments/gpu-sim-u2o.17.5/canonical_fixture.json"

RTX_5090_FP32_TFLOPS = 104.8e12
RTX_5090_BW_BYTES_S = 1568e9


def case_size(c):
    s = c["shapes"]
    if c["op_class"] == "matmul":
        n = s[0]
        return 2 * n ** 3  # flops
    p = 1
    for d in s:
        p *= d
    return p  # elements


def case_bytes(c):
    s = c["shapes"]
    if c["op_class"] == "matmul":
        n = s[0]
        return 3 * n * n * 4
    p = 1
    for d in s:
        p *= d
    return 3 * p * 4  # read+read+write fp32


def naive_us(c):
    if c["op_class"] == "matmul":
        return case_size(c) / RTX_5090_FP32_TFLOPS * 1e6
    return case_bytes(c) / RTX_5090_BW_BYTES_S * 1e6


def loglog_fit(pts):
    xs = [math.log(x) for x, _ in pts]
    ys = [math.log(y) for _, y in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return my - b * mx, b


def main(out=None) -> int:
    out = Path(out) if out else (REPO / "experiments/gpu-sim-u2o.17.3"
                                 / "metrics.json")
    fix = json.loads(FIX.read_text())
    # size point per (class, shapes): median over same-size cases
    med = {}
    reps = {}
    for c in fix["cases"]:
        med.setdefault((c["op_class"], str(c["shapes"])), []).append(
            (c["device_ms_median"], c))
    per_cls = {}
    for (cls, sh), lst in med.items():
        vs = sorted(v for v, _ in lst)
        per_cls.setdefault(cls, []).append(
            (size_of(cls, sh), sh, vs[len(vs) // 2],
             lst[0][1]))
    result = {"bead": "gpu-sim-u2o.17.3",
              "fixture": "experiments/gpu-sim-u2o.17.5/"
                           "canonical_fixture.json",
              "method": "leave-one-size-out log-log fit per op class; "
                          "naive roofline on same held-out cases",
              "classes": {}, "gaps": []}
    for cls, points in sorted(per_cls.items()):
        points.sort()
        if len(points) < 3:
            result["gaps"].append(
                f"{cls}: only {len(points)} sizes, no fit (needs >=3)")
            continue
        errs_fit, errs_naive = [], []
        detail = []
        for i, (sz, sh, med_ms, rep) in enumerate(points):
            train = [(s2, m2) for j, (s2, _s, m2, _r) in enumerate(points)
                     if j != i]
            a, b = loglog_fit(train)
            pred_ms = math.exp(a + b * math.log(sz))
            naive_ms = naive_us(rep) / 1000.0
            ef = (pred_ms - med_ms) / med_ms
            en = (naive_ms - med_ms) / med_ms
            errs_fit.append(abs(ef))
            errs_naive.append(abs(en))
            detail.append({"shapes": sh, "measured_ms": med_ms,
                           "fit_pred_ms": pred_ms,
                           "fit_rel_err": ef, "naive_rel_err": en})
        result["classes"][cls] = {
            "n_sizes": len(points),
            "mean_abs_err_fit": sum(errs_fit) / len(errs_fit),
            "mean_abs_err_naive": sum(errs_naive) / len(errs_naive),
            "held_out": detail,
        }
    out.write_text(json.dumps(result, indent=2))
    for cls, r in result["classes"].items():
        print(f"  {cls}: fit MAE {r['mean_abs_err_fit']:.3f} vs naive "
              f"{r['mean_abs_err_naive']:.3f} (n={r['n_sizes']})")
    print("gaps:", result["gaps"])
    return 0


def size_of(cls, sh):
    s = json.loads(sh)
    if cls == "matmul":
        return 2 * s[0] ** 3
    p = 1
    for d in s:
        p *= d
    return p


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
