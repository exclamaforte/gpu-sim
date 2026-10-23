import json, math, statistics
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
CAL = json.loads((REPO/"experiments/gpu-sim-m93/mdz_calibration_fixture.json").read_text())
FIX = json.loads((Path(__file__).parent/"v4e_fixture.json").read_text())

# build calibration tables per (op_class, op) keyed by shapes[0]
from collections import defaultdict
cal_pts = defaultdict(lambda: defaultdict(list))
for c in CAL["cases"]:
    if "error" in c: continue
    shapes = c["shapes"]
    if isinstance(shapes[0], list): shapes = shapes[0]
    sz = shapes[0]
    op = c.get("op", c["op_class"])
    med = c["device_ms_median"]
    cal_pts[(c["op_class"], op)][sz].append(med)
cal_tables = {}
for key, bysz in cal_pts.items():
    cal_tables[key] = sorted((sz, statistics.median(meds)) for sz, meds in bysz.items())

def interp(table, x):
    table = sorted(table)
    if x <= table[0][0]: return table[0][1], table[0][0]
    if x >= table[-1][0]: return table[-1][1], table[-1][0]
    for i in range(len(table)-1):
        if table[i][0] <= x <= table[i+1][0]:
            x0,y0=table[i]; x1,y1=table[i+1]
            t=(math.log(x)-math.log(x0))/(math.log(x1)-math.log(x0))
            return math.exp(math.log(y0)+t*(math.log(y1)-math.log(y0))), None
    return table[-1][1], table[-1][0]

results = []
for c in FIX["cases"]:
    cls = c["op_class"]; op = c["op"]; shapes = c["shapes"]
    if isinstance(shapes[0], list): shapes = shapes[0]
    sz = shapes[0]
    key = (cls, op)
    if key not in cal_tables:
        keys = [k for k in cal_tables if k[0]==cls]
        key = keys[0]
    table = cal_tables[key]
    cal_min = table[0][0]; cal_max = table[-1][0]
    pred, clamp_sz = interp(table, sz)
    meas = c["device_ms_median"]
    err = (pred - meas)/meas
    clamp_target = None
    if sz < cal_min: clamp_target = cal_min
    elif sz > cal_max: clamp_target = cal_max
    results.append({"op_class":cls,"op":op,"size":sz,"measured_ms":meas,"pred_ms":pred,
                    "rel_err":err,"abs_rel_err":abs(err),"cal_min":cal_min,"cal_max":cal_max,
                    "clamp_target":clamp_target,"direction": "below" if sz<cal_min else ("above" if sz>cal_max else "in-grid")})
    print(f"{cls:12} {op:8} size={sz:>12} meas={meas:.4f} pred={pred:.4f} err={err:+.3f} clamp={clamp_target}")

# envelope summary
for cls in ["elementwise","reduce","matmul"]:
    below = sorted([r for r in results if r["op_class"]==cls and r["direction"]=="below"], key=lambda r:r["size"], reverse=True)
    above = sorted([r for r in results if r["op_class"]==cls and r["direction"]=="above"], key=lambda r:r["size"])
    print("\n", cls)
    print("  below (largest first):")
    for r in below:
        octs = math.log2(r["cal_min"]/r["size"])
        print(f"    size={r['size']:>12} octs_below={octs:.2f} err={r['abs_rel_err']:.3f} {'OVER' if r['abs_rel_err']>0.25 else 'ok'}")
    print("  above (smallest first):")
    for r in above:
        octs = math.log2(r["size"]/r["cal_max"])
        print(f"    size={r['size']:>12} octs_above={octs:.2f} err={r['abs_rel_err']:.3f} {'OVER' if r['abs_rel_err']>0.25 else 'ok'}")

out = Path(__file__).parent / "v4e_scored.json"
out.write_text(json.dumps({"results":results}, indent=2))
print("\nwrote", out)
