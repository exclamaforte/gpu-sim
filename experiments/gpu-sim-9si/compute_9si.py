#!/usr/bin/env python3
import json, math, statistics, pathlib
HERE=pathlib.Path(__file__).resolve().parent
REPO=HERE.parents[1]
v1=json.loads((REPO/"experiments/gpu-sim-u2o.17.5/canonical_fixture.json").read_text())
v2=json.loads((HERE/"fixture_v2.json").read_text())
mdz=json.loads((HERE/"mdz_calibration_fixture.json").read_text())

from collections import defaultdict
by_op=defaultdict(lambda: defaultdict(list))
for c in mdz["cases"]:
    by_op[(c["op_class"],c["op"])][c["shapes"][0] if not isinstance(c["shapes"][0],list) else c["shapes"][0][0] if isinstance(c["shapes"][0],list) else c["shapes"][0]].append(c["device_ms_median"])
# simplify shape extraction
by_op=defaultdict(lambda: defaultdict(list))
for c in mdz["cases"]:
    sh=c["shapes"][0]
    if isinstance(sh,list): sh=sh[0]
    by_op[(c["op_class"],c["op"])][sh].append(c["device_ms_median"])
op_tables={k:sorted((sz,statistics.median(meds)) for sz,meds in d.items()) for k,d in by_op.items()}

def interp(table,x):
    table=sorted(table)
    if x<=table[0][0]: return table[0][1]
    if x>=table[-1][0]: return table[-1][1]
    for i in range(len(table)-1):
        if table[i][0]<=x<=table[i+1][0]:
            x0,y0=table[i]; x1,y1=table[i+1]
            t=(math.log(x)-math.log(x0))/(math.log(x1)-math.log(x0))
            return math.exp(math.log(y0)+t*(math.log(y1)-math.log(y0)))
    return table[-1][1]

# group v1 and v2 by (op_class, tuple(shapes))
def group(fix):
    g=defaultdict(list)
    for c in fix["cases"]:
        g[(c["op_class"], c["op"], tuple(c["shapes"]))].append(c)
    return g
g1=group(v1); g2=group(v2)

per_case=[]
for key,lst1 in g1.items():
    cls=key[0]
    op=key[1]
    if cls not in ("elementwise","matmul","reduce"): continue
    lst2=g2.get(key,[])
    if not lst2: 
        continue
    meas1=statistics.median(c["device_ms_median"] for c in lst1)
    meas2=statistics.median(c["device_ms_median"] for c in lst2)
    drift=abs(meas2-meas1)/meas1
    # op from first case
    op=key[1]
    n=key[2][0]
    # x for interp: for matmul use n, else n
    x=n
    table=op_tables.get((cls,op))
    if table is None:
        # fallback to first op in class
        table=op_tables[[k for k in op_tables if k[0]==cls][0]]
    pred_ms=interp(table,x)
    err1=abs((pred_ms-meas1)/meas1)
    err2=abs((pred_ms-meas2)/meas2)
    per_case.append({"op_class":cls,"op":op,"shapes":list(key[2]),
                    "measured_v1_ms":meas1,"measured_v2_ms":meas2,
                    "drift":drift,"pred_ms":pred_ms,
                    "err_v1":err1,"err_v2":err2,
                    "err_shift":abs(err2-err1)})

# per class
per_class={}
for cls in ["elementwise","matmul","reduce"]:
    cs=[p for p in per_case if p["op_class"]==cls]
    per_class[cls]={"median_err_v1":statistics.median(p["err_v1"] for p in cs),
                    "median_err_v2":statistics.median(p["err_v2"] for p in cs),
                    "max_drift":max(p["drift"] for p in cs),
                    "median_drift":statistics.median(p["drift"] for p in cs),
                    "n":len(cs)}

# verdict
medians_v2=[per_class[cls]["median_err_v2"] for cls in per_class]
max_shift=max(abs(p["err_v2"]-p["err_v1"]) for p in per_case)
if any(m>0.25 for m in medians_v2):
    verdict="SESSION-FRAGILE"
elif max_shift>0.10:
    verdict="DRIFT-LIMITED"
else:
    verdict="STABLE"

cross_check={
 "m93_committed_max_err": {"elementwise":0.10391816733791204,"matmul":0.17473286607405988,"reduce":0.19554071223927577},
 "recomputed_max_err": {cls: max(p["err_v1"] for p in per_case if p["op_class"]==cls) for cls in ["elementwise","matmul","reduce"]},
 "match": all(abs(max(p["err_v1"] for p in per_case if p["op_class"]==cls) - {"elementwise":0.10391816733791204,"matmul":0.17473286607405988,"reduce":0.19554071223927577}[cls])<1e-9 for cls in ["elementwise","matmul","reduce"]),
}
metrics={
 "bead":"gpu-sim-9si",
 "table_sha256_note":"99b0e78b (mdz table, unchanged)",
 "n_scored_cases":len(per_case),
 "per_case":per_case,
 "per_class":per_class,
 "max_err_shift":max_shift,
 "verdict":verdict,
 "cross_check_m93":cross_check,
}
HERE.joinpath("metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
print("verdict",verdict)
print(json.dumps(per_class,indent=2))
