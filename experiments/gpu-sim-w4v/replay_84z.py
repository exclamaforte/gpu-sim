#!/usr/bin/env python3
import json, os, pathlib, statistics
HERE=pathlib.Path(__file__).resolve().parent
os.environ["GPUSIM_OP_LATENCY_JSON"]=str(HERE/"mdz_calibration_fixture.json")
import sys; sys.path.insert(0,str(HERE.parents[1]))
from epic_u2o.op_latency import latency_ms_for
from epic_u2o import op_latency
op_latency._cache.clear()
dtype=json.loads((HERE/"dtype_fixture.json").read_text())
errs=[]
for c in dtype["cases"]:
    if c["dtype"]!="float16": continue
    n=c["shapes"][0]; meas=c["device_ms_median"]
    pred=latency_ms_for(c["op"], n, dtype_bytes=2)
    errs.append(abs((pred-meas)/meas))
    print(c["op"], n, "measured", meas, "pred", pred, "err", round(errs[-1],4))
print("median", statistics.median(errs), "max", max(errs), "n", len(errs))
metrics={"bead":"gpu-sim-w4v","replay":"84z fp16","n":len(errs),
        "median_err":statistics.median(errs),"max_err":max(errs),
        "all_pass": all(e<=0.25 for e in errs)}
HERE.joinpath("replay_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
