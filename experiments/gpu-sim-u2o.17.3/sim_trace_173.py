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
import statistics
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


def main(out=None, extra=None, calibration=None) -> int:
    out = Path(out) if out else (REPO / "experiments/gpu-sim-u2o.17.3"
                                 / "metrics.json")
    fix = json.loads(FIX.read_text())
    cases = list(fix["cases"])
    if extra and not calibration:
        sup = json.loads(Path(extra).read_text())
        cases += sup["cases"]
        print(f"merged {sup['n_cases']} supplement cases")
    if calibration:
        cal = json.loads(Path(calibration).read_text())
        cal_cases = cal["cases"] if "cases" in cal else cal.get("per_case", [])
        # build calibration points per (op_class, op)
        from collections import defaultdict
        cal_pts = defaultdict(lambda: defaultdict(list))
        for c in cal_cases:
            if "error" in c:
                continue
            shapes = c.get("shapes", [c.get("shape")])
            if isinstance(shapes[0], list):
                shapes = shapes[0]
            sz = shapes[0]
            op = c.get("op", c["op_class"])
            if "device_ms_median" in c:
                med = c["device_ms_median"]
            elif "median_us" in c:
                med = c["median_us"] / 1000.0
            else:
                continue
            cal_pts[(c["op_class"], op)][sz].append(med)
        cal_tables = {}
        for key, bysz in cal_pts.items():
            cal_tables[key] = sorted((sz, statistics.median(meds)) for sz, meds in bysz.items())
        # disjointness assertion
        fixture_shapes = set((c["op_class"], tuple(c["shapes"])) for c in cases)
        for c in cal_cases:
            assert (c["op_class"], tuple(c["shapes"])) not in fixture_shapes, \
                f"calibration overlaps scored fixture: {c['op_class']} {c['shapes']}"
        print("disjointness: OK (no calibration shape in scored set)")

        def interp(table, x):
            table = sorted(table)
            if not table:
                raise RuntimeError("empty calibration table")
            if x <= table[0][0]:
                return table[0][1]
            if x >= table[-1][0]:
                return table[-1][1]
            for i in range(len(table)-1):
                if table[i][0] <= x <= table[i+1][0]:
                    x0,y0=table[i]; x1,y1=table[i+1]
                    t=(math.log(x)-math.log(x0))/(math.log(x1)-math.log(x0))
                    return math.exp(math.log(y0)+t*(math.log(y1)-math.log(y0)))
            return table[-1][1]

        result = {"bead": "gpu-sim-u2o.17.3",
                  "method": "external calibration table, log-log interpolation",
                  "calibration": str(Path(calibration).relative_to(REPO) if Path(calibration).is_absolute() and Path(calibration).is_relative_to(REPO) else calibration),
                  "calibration_sha256": None,
                  "classes": {}, "gaps": [], "disjointness": None}
        import hashlib
        result["calibration_sha256"] = hashlib.sha256(Path(calibration).read_bytes()).hexdigest()
        result["disjointness"] = {"n_calibration_cases": len(cal_cases), "ok": True}
        # per-class aggregation
        per_cls = defaultdict(list)
        for c in cases:
            per_cls[c["op_class"]].append(c)
        for cls in sorted(per_cls):
            if not any(k[0]==cls for k in cal_tables):
                result["gaps"].append(f"{cls}: no calibration table")
                continue
            points = []
            detail = []
            for c in per_cls[cls]:
                sh = str(c["shapes"])
                sz = c["shapes"][0] if cls != "matmul" else c["shapes"][0]
                if cls == "matmul":
                    sz = c["shapes"][0]
                op = c.get("op", cls)
                key = (cls, op)
                if key not in cal_tables:
                    # fallback to any op in class
                    keys = [k for k in cal_tables if k[0]==cls]
                    if not keys:
                        result["gaps"].append(f"{cls}: no calibration table")
                        continue
                    key = keys[0]
                table = cal_tables[key]
                pred_ms = interp(table, sz) / 1000.0 if False else None
                # table values are ms? calibration device_ms_median in ms. Store ms.
                pred_ms = interp(table, sz)
                measured_ms = c["device_ms_median"]
                ef = (pred_ms - measured_ms) / measured_ms
                detail.append({"shapes": c["shapes"], "measured_ms": measured_ms,
                               "fit_pred_ms": pred_ms, "fit_rel_err": ef})
                points.append((sz, measured_ms))
            if len(detail) < 1:
                result["gaps"].append(f"{cls}: no cases")
                continue
            result["classes"][cls] = {"n_sizes": len(detail),
                                     "held_out": detail}
        out_path = Path(out) if out else None
        if out_path:
            out_path.write_text(json.dumps(result, indent=2))
        for cls, r in result["classes"].items():
            errs = [abs(d["fit_rel_err"]) for d in r["held_out"]]
            med = statistics.median(errs)
            print(f"  {cls}: median err {med:.3f} (n={r['n_sizes']})")
        return 0

    # size point per (class, shapes): median over same-size cases
    med = {}
    reps = {}
    for c in cases:
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
    a = sys.argv[1:]
    out = a[0] if a and not a[0].startswith("--") else None
    extra = None
    calibration = None
    for i, x in enumerate(a):
        if x == "--extra" and i + 1 < len(a):
            extra = a[i + 1]
        if x == "--calibration" and i + 1 < len(a):
            calibration = a[i + 1]
    raise SystemExit(main(out, extra, calibration))
