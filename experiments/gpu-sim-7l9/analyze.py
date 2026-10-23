#!/usr/bin/env python3
import json, math, statistics, pathlib
HERE=pathlib.Path(__file__).resolve().parent
REPO=HERE.parents[1]
mdz=json.loads((REPO/"experiments/gpu-sim-mdz/mdz_calibration_fixture.json").read_text())
unseen=json.loads((HERE/"unseen_fixture.json").read_text())

def interp(table, x):
    table=sorted(table)
    if x<=table[0][0]: return table[0][1]
    if x>=table[-1][0]: return table[-1][1]
    for i in range(len(table)-1):
        if table[i][0]<=x<=table[i+1][0]:
            x0,y0=table[i]; x1,y1=table[i+1]
            t=(math.log(x)-math.log(x0))/(math.log(x1)-math.log(x0))
            return math.exp(math.log(y0)+t*(math.log(y1)-math.log(y0)))
    return table[-1][1]

# build per-op tables
from collections import defaultdict
by_op=defaultdict(lambda: defaultdict(list))
for c in mdz["cases"]:
    if c["op_class"]!="elementwise": continue
    by_op[c["op"]][c["shapes"][0]].append(c["device_ms_median"])
op_tables={}
for op,d in by_op.items():
    op_tables[op]=sorted((sz, statistics.median(meds)) for sz,meds in d.items())
# class-median table: median across ops per size
by_size=defaultdict(list)
for c in mdz["cases"]:
    if c["op_class"]!="elementwise": continue
    by_size[c["shapes"][0]].append(c["device_ms_median"])
class_table=sorted((sz, statistics.median(meds)) for sz,meds in by_size.items())

results=[]
for c in unseen["cases"]:
    n=c["shapes"][0]; meas=c["device_ms_median"]
    # nearest per-op: relu
    relu_pred=interp(op_tables["relu"], n)/1000.0  # table values ms? device_ms_median is ms, so interp ms
    # Wait table values are device_ms_median (ms). interp returns ms. Convert to ms? keep ms.
    relu_pred_ms=interp(op_tables["relu"], n)
    class_pred_ms=interp(class_table, n)
    relu_err=abs((relu_pred_ms-meas)/meas)
    class_err=abs((class_pred_ms-meas)/meas)
    results.append({"op":c["op"],"size":n,"measured_ms":meas,"relu_pred_ms":relu_pred_ms,"class_pred_ms":class_pred_ms,"relu_err":relu_err,"class_err":class_err})

# per-case table
for r in results:
    print(r["op"], r["size"], "measured", round(r["measured_ms"],4), "relu_pred", round(r["relu_pred_ms"],4), "relu_err", round(r["relu_err"],4), "class_err", round(r["class_err"],4))

# per-op summary
for op in ["gelu","hardsigmoid"]:
    rs=[r for r in results if r["op"]==op]
    print(op, "relu-pred median", statistics.median(r["relu_err"] for r in rs), "max", max(r["relu_err"] for r in rs))
    print(op, "class-pred median", statistics.median(r["class_err"] for r in rs), "max", max(r["class_err"] for r in rs))

# by size band
for band in [1<<10,1<<16,1<<20,1<<24]:
    rs=[r for r in results if r["size"]==band]
    print("size",band,"relu median", statistics.median(r["relu_err"] for r in rs),"class median", statistics.median(r["class_err"] for r in rs))

out={"results":results,
     "summary":{
       "gelu":{"relu_median":statistics.median(r["relu_err"] for r in results if r["op"]=="gelu"),
               "relu_max":max(r["relu_err"] for r in results if r["op"]=="gelu"),
               "class_median":statistics.median(r["class_err"] for r in results if r["op"]=="gelu"),
               "class_max":max(r["class_err"] for r in results if r["op"]=="gelu")},
       "hardsigmoid":{"relu_median":statistics.median(r["relu_err"] for r in results if r["op"]=="hardsigmoid"),
                      "relu_max":max(r["relu_err"] for r in results if r["op"]=="hardsigmoid"),
                      "class_median":statistics.median(r["class_err"] for r in results if r["op"]=="hardsigmoid"),
                      "class_max":max(r["class_err"] for r in results if r["op"]=="hardsigmoid")},
     }}
HERE.joinpath("analysis.json").write_text(json.dumps(out,indent=2)+"\n")
print("wrote analysis.json")
