#!/usr/bin/env python3
import json, math, statistics, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"experiments/gpu-sim-u2o.17.3"))

FIX = REPO / "experiments/gpu-sim-u2o.17.5/canonical_fixture.json"
KNEE = REPO / "experiments/gpu-sim-u2o.17.3/knee_cases.json"
DISPATCH_FLOOR_US = 1.0016
BW_BYTES_S = 1688e9
PEAK_TFLOPS = 82.6e12

def case_size(c):
    s=c["shapes"]
    if c["op_class"]=="matmul":
        return s[0]
    p=1
    for d in s: p*=d
    return p

def case_bytes(c):
    s=c["shapes"]
    if c["op_class"]=="matmul":
        n=s[0]; return 3*n*n*4
    p=1
    for d in s: p*=d
    if c["op_class"]=="reduce":
        return 2*p*4
    return 3*p*4

def knee_eff_bw():
    d=json.loads(KNEE.read_text())
    bws=[]
    for c in d["cases"]:
        n=case_size(c); b=case_bytes(c); t=c["device_ms_median"]/1000.0
        bws.append(b/t)
    return statistics.median(bws)

eff_bw = knee_eff_bw()

def physical_pred_us(c):
    n=case_size(c); b=case_bytes(c)
    t_us = DISPATCH_FLOOR_US
    if c["op_class"]=="matmul":
        flops=2*n**3
        t_us = max(t_us, flops/PEAK_TFLOPS*1e6, b/BW_BYTES_S*1e6)
    else:
        t_us = max(t_us, b/eff_bw*1e6)
    return t_us

fix=json.loads(FIX.read_text())
cases=fix["cases"]
from collections import defaultdict
groups=defaultdict(list)
for c in cases:
    groups[(c["op_class"],tuple(c["shapes"]))].append(c)

noise={}
for (cls,sh),lst in groups.items():
    meds=[c["device_ms_median"] for c in lst]
    if len(meds)>=2:
        mx=max(meds); mn=min(meds); med=statistics.median(meds)
        spread=(mx-mn)/med
    else:
        spread=0.0
    noise.setdefault(cls,[]).append((list(sh), len(meds), spread))

per_class_err_lever3={}
for cls in ["elementwise","matmul","reduce"]:
    errs=[]
    for (ccls,sh),lst in groups.items():
        if ccls!=cls: continue
        med_ms=statistics.median(c["device_ms_median"] for c in lst)
        pred_us=physical_pred_us(lst[0])
        pred_ms=pred_us/1000.0
        errs.append(abs((pred_ms-med_ms)/med_ms))
    per_class_err_lever3[cls]=statistics.median(errs) if errs else None

knee=json.loads(KNEE.read_text())
knee_shapes=set((c["op_class"],tuple(c["shapes"])) for c in knee["cases"])
fixture_shapes=set((c["op_class"],tuple(c["shapes"])) for c in cases)
overlap=sorted(knee_shapes & fixture_shapes)

high_spread=[]
for (cls,sh),lst in groups.items():
    meds=[c["device_ms_median"] for c in lst]
    if len(meds)>=2:
        mx=max(meds); mn=min(meds); med=statistics.median(meds)
        spread=(mx-mn)/med
        if spread>0.25:
            high_spread.append({"op_class":cls,"shapes":list(sh),"spread":round(spread,4),"n_repeats":len(meds)})

# elementwise error by size band
ew_bands={}
for (ccls,sh),lst in groups.items():
    if ccls!="elementwise": continue
    n=sh[0]
    band = "<=65536" if n<=65536 else ("1048576" if n==1048576 else "16777216")
    med_ms=statistics.median(c["device_ms_median"] for c in lst)
    pred_ms=physical_pred_us(lst[0])/1000.0
    ew_bands.setdefault(band,[]).append(abs((pred_ms-med_ms)/med_ms))
ew_band_medians={b: (statistics.median(v) if v else None) for b,v in ew_bands.items()}
metrics={
  "bead":"gpu-sim-gh9",
  "calibration_source":"knee_cases.json (elementwise only)",
  "constants":{
    "dispatch_floor_us":{"value":DISPATCH_FLOOR_US,"source":"same-trace launch_bound case, gpu-sim-u2o.17.5 (256 elements, 0.0010016 ms, spread 3.2%)"},
    "effective_bandwidth_bytes_s":{"value":eff_bw,"source":"median of knee_cases elementwise bytes/time"},
    "peak_tflops":{"value":PEAK_TFLOPS/1e12,"source":"hw_5090 vendor spec (unsourced in-tree), used only for matmul"},
    "memory_bandwidth_bytes_s":{"value":BW_BYTES_S,"source":"hw_5090 measured read bw"},
  },
  "shape_overlap_knee_vs_fixture": [list(x) for x in overlap],
  "noise_floor":{
    "per_class_median_spread": {cls: statistics.median([s for (_,_,s) in noise[cls]]) if cls in noise else None for cls in ["elementwise","matmul","reduce"]},
    "per_class_max_spread": {cls: max([s for (_,_,s) in noise[cls]]) if cls in noise else None for cls in ["elementwise","matmul","reduce"]},
    "high_spread_cases": high_spread,
  },
  "baseline": {"elementwise":0.969,"matmul":0.409,"reduce":0.473},
  "lever3": per_class_err_lever3,
  "lever3_elementwise_by_band": ew_band_medians,
  "residual_assessment": "After item1 (correct floor), elementwise residual by band: see lever3_elementwise_by_band. Regime structure remains (floor-bound small, L2 mid, cache-degraded large); not fixable with a single constant.",
  "verdict": None,
  "bar": 0.25,
  "bar_criterion": "u2o.17: median |sim-measured|/measured <= 25% per op class",
}
passes=sum(1 for c in ["elementwise","matmul","reduce"] if per_class_err_lever3[c]<=0.25)
if passes==3: metrics["verdict"]="PASS-ALL"
elif passes>=1: metrics["verdict"]="PARTIAL"
else: metrics["verdict"]="GAP-REMAINS"
pathlib.Path("experiments/gpu-sim-gh9/metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
print(json.dumps(metrics,indent=2))
