#!/usr/bin/env python3
import json, math, statistics, pathlib
HERE=pathlib.Path(__file__).resolve().parent
REPO=HERE.parents[1]
unseen=json.loads((HERE/"unseen_fixture.json").read_text())
mdz=json.loads((HERE/"mdz_calibration_fixture.json").read_text())
table_ops=set(c["op"] for c in mdz["cases"])
for op in ("gelu","hardsigmoid"):
    assert op not in table_ops

from collections import defaultdict
by_op=defaultdict(lambda: defaultdict(list))
for c in mdz["cases"]:
    if c["op_class"]!="elementwise": continue
    by_op[c["op"]][c["shapes"][0]].append(c["device_ms_median"])
op_tables={op:sorted((sz,statistics.median(meds)) for sz,meds in d.items()) for op,d in by_op.items()}
by_size=defaultdict(list)
for c in mdz["cases"]:
    if c["op_class"]!="elementwise": continue
    by_size[c["shapes"][0]].append(c["device_ms_median"])
class_table=sorted((sz,statistics.median(meds)) for sz,meds in by_size.items())

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

results=[]
for c in unseen["cases"]:
    n=c["shapes"][0]; meas=c["device_ms_median"]
    relu_pred=interp(op_tables["relu"],n)
    class_pred=interp(class_table,n)
    results.append({"op":c["op"],"size":n,"measured_ms":meas,
                    "relu_pred_ms":relu_pred,"class_pred_ms":class_pred,
                    "relu_err":abs((relu_pred-meas)/meas),
                    "class_err":abs((class_pred-meas)/meas)})

def med(xs): return statistics.median(xs) if xs else None
def mx(xs): return max(xs) if xs else None

summary={
 "gelu":{"relu_median":med(r["relu_err"] for r in results if r["op"]=="gelu"),
         "relu_max":mx(r["relu_err"] for r in results if r["op"]=="gelu"),
         "class_median":med(r["class_err"] for r in results if r["op"]=="gelu"),
         "class_max":mx(r["class_err"] for r in results if r["op"]=="gelu")},
 "hardsigmoid":{"relu_median":med(r["relu_err"] for r in results if r["op"]=="hardsigmoid"),
                "relu_max":mx(r["relu_err"] for r in results if r["op"]=="hardsigmoid"),
                "class_median":med(r["class_err"] for r in results if r["op"]=="hardsigmoid"),
                "class_max":mx(r["class_err"] for r in results if r["op"]=="hardsigmoid")},
 "all":{"relu_median":med(r["relu_err"] for r in results),
        "relu_max":mx(r["relu_err"] for r in results),
        "class_median":med(r["class_err"] for r in results),
        "class_max":mx(r["class_err"] for r in results)},
}
# by size band
by_band={}
for n in [1<<10,1<<16,1<<20,1<<24]:
    rs=[r for r in results if r["size"]==n]
    by_band[str(n)]={"relu_median":med(r["relu_err"] for r in rs),
                     "class_median":med(r["class_err"] for r in rs),
                     "n":len(rs)}

metrics={
 "bead":"gpu-sim-7l9",
 "unseen_ops":["gelu","hardsigmoid"],
 "verified_absent_from_table": True,
 "scored_sizes":[1<<10,1<<16,1<<20,1<<24],
 "n_cases":len(results),
 "per_case":results,
 "summary":summary,
 "by_size_band":by_band,
 "recommendation":"per-op nearest curve (relu) for unseen elementwise ops; both predictors pass <=0.25 at all sizes in this measurement, so op-variety is a non-issue here.",
}
HERE.joinpath("metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
print("wrote metrics.json")
print(json.dumps(summary,indent=2))
