#!/usr/bin/env python3
import json, statistics, pathlib, sys, math
from collections import defaultdict
HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path("/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-mdz")
FIX = HERE / "calibration_fixture.json"
d = json.loads(FIX.read_text())
cases = d["cases"]

def bytes_for(c):
    if c["op_class"] == "matmul":
        n=c["shapes"][0]; return 3*n*n*4
    n = c["shapes"][0]
    if c["op_class"] == "reduce":
        return 2*n*4
    return 3*n*4

rows = []
for c in cases:
    if "error" in c:
        continue
    n = c["shapes"][0] if c["op_class"]!="matmul" else c["shapes"][0]
    b = bytes_for(c)
    t = c["device_ms_median"] / 1000.0
    bw = b / t
    rows.append({"op_class": c["op_class"], "op": c["op"], "shape": c["shapes"],
                "median_us": c["device_ms_median"]*1000.0,
                "spread_pct": c.get("device_ms_spread_pct_of_median"),
                "bytes": b, "implied_bw_bytes_s": bw})

# fixed cost by op from smallest two sizes
fixed = {}
for cls in ["elementwise", "reduce", "launch_bound", "matmul"]:
    clsrows = [r for r in rows if r["op_class"] == cls]
    if not clsrows:
        continue
    clsrows_sorted = sorted(clsrows, key=lambda r: (r["shape"][0] if cls!="matmul" else r["shape"][0]))
    small = clsrows_sorted[:2]
    medbw = statistics.median(r["implied_bw_bytes_s"] for r in clsrows)
    floors = []
    for r in small:
        floor_us = r["median_us"] - r["bytes"]/medbw*1e6
        floors.append(floor_us)
    fixed[cls] = {"smallest_two_sizes": [r["shape"] for r in small],
                  "median_bw_bytes_s": medbw,
                  "fixed_cost_us_estimates": floors,
                  "fixed_cost_us_median": statistics.median(floors)}

spread_summary = {}
for cls in ["elementwise", "reduce", "launch_bound", "matmul"]:
    sps = [r["spread_pct"] for r in rows if r["op_class"]==cls and r["spread_pct"] is not None]
    if sps:
        spread_summary[cls] = {"median_spread_pct": statistics.median(sps),
                               "max_spread_pct": max(sps), "n": len(sps)}

# prediction test: log-log interpolation of calibration medians per op class -> fixture scored cases
calib_medians = {}
for cls in ["elementwise", "matmul", "reduce"]:
    for op in sorted(set(r["op"] for r in rows if r["op_class"]==cls)):
        pts = sorted([(r["shape"][0], r["median_us"]) for r in rows if r["op_class"]==cls and r["op"]==op])
        bysize = {}
        for sz, med in pts:
            bysize.setdefault(sz, []).append(med)
        calib_medians[(cls,op)] = sorted((sz, statistics.median(meds)) for sz, meds in bysize.items())

def interp(calib, x):
    calib = sorted(calib)
    if x <= calib[0][0]:
        return calib[0][1]
    if x >= calib[-1][0]:
        return calib[-1][1]
    for i in range(len(calib)-1):
        if calib[i][0] <= x <= calib[i+1][0]:
            x0,y0=calib[i]; x1,y1=calib[i+1]
            t=(math.log(x)-math.log(x0))/(math.log(x1)-math.log(x0))
            return math.exp(math.log(y0)+t*(math.log(y1)-math.log(y0)))
    return calib[-1][1]

fixture = json.loads((REPO/"experiments/gpu-sim-u2o.17.5/canonical_fixture.json").read_text())

pred_errors = defaultdict(list)
per_case = []
for c in fixture["cases"]:
    cls = c["op_class"]
    if cls not in ["elementwise","matmul","reduce"]:
        continue
    sh = c["shapes"]
    med_ms = c["device_ms_median"]
    n = sh[0]
    x = n
    op = c["op"]
    key = (cls, op)
    if key not in calib_medians:
        key = (cls, sorted(set(r["op"] for r in rows if r["op_class"]==cls))[0])
    pred_us = interp(calib_medians[key], x)
    pred_ms = pred_us/1000.0
    err = abs((pred_ms-med_ms)/med_ms)
    pred_errors[cls].append(err)
    per_case.append({"op_class":cls,"op":c["op"],"shapes":list(sh),"measured_ms":med_ms,"pred_ms":pred_ms,"rel_err":err})

prediction_test = {
    "per_class_median_rel_err": {cls: (statistics.median(pred_errors[cls]) if pred_errors[cls] else None) for cls in ["elementwise","matmul","reduce"]},
    "per_class_max_rel_err": {cls: (max(pred_errors[cls]) if pred_errors[cls] else None) for cls in ["elementwise","matmul","reduce"]},
    "per_case": per_case,
    "old_numbers": {"elementwise":0.969,"matmul":0.409,"reduce":0.473},
}

# across-invocation spread
inv_files = [HERE/"calibration_fixture.json", HERE/"calibration_fixture_inv2.json", HERE/"calibration_fixture_inv3.json"]
inv_data = [json.loads(f.read_text()) for f in inv_files if f.exists()]
# map (op_class, tuple(shapes)) -> list of medians
from collections import defaultdict
inv_map = defaultdict(list)
for dd in inv_data:
    for c in dd["cases"]:
        if "error" in c: continue
        inv_map[(c["op_class"], tuple(c["shapes"]))].append(c["device_ms_median"])
across = []
for key, meds in inv_map.items():
    if len(meds) < 2: continue
    mx=max(meds); mn=min(meds); med=statistics.median(meds)
    spread=(mx-mn)/med
    across.append({"op_class":key[0],"shapes":list(key[1]),"medians_us":[m*1000 for m in meds],"spread":spread})
across_spreads=[a["spread"] for a in across]
high_across=[a for a in across if a["spread"]>0.25]
across_summary={"median_spread": statistics.median(across_spreads) if across_spreads else None,
                "max_spread": max(across_spreads) if across_spreads else None,
                "n_cases": len(across),
                "high_spread_cases": high_across}
metrics = {
    "bead": "gpu-sim-mdz",
    "calibration_fixture": "experiments/gpu-sim-mdz/calibration_fixture.json",
    "n_cases": len(rows),
    "byte_model": "elementwise 3*p*4, reduce 2*p*4, matmul 3*n*n*4",
    "per_case": rows,
    "fixed_cost_by_op": fixed,
    "spread_summary": spread_summary,
    "invocations": len(inv_data),
    "gpu_minutes": 30*len(inv_data),
    "across_invocation_spread": across_summary,
    "prediction_test": prediction_test,
    "disjointness": "calibration sizes exclude scored sizes (asserted in capture script)",
}
HERE.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
print("wrote metrics.json with", len(rows), "cases")
print("prediction medians:", prediction_test["per_class_median_rel_err"])
