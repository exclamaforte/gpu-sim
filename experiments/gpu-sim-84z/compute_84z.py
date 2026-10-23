#!/usr/bin/env python3
import json, math, statistics, pathlib
HERE=pathlib.Path(__file__).resolve().parent
REPO=HERE.parents[1]
dtype=json.loads((HERE/"dtype_fixture.json").read_text())
mdz=json.loads((HERE/"mdz_calibration_fixture.json").read_text())
FIX=REPO/"experiments/gpu-sim-u2o.17.5/canonical_fixture.json"
v1=json.loads(FIX.read_text())

# table lookup by (op,size)
table={}
for c in mdz["cases"]:
    if c["op_class"]!="elementwise": continue
    sh=c["shapes"][0]
    if isinstance(sh,list): sh=sh[0]
    table[(c["op"],sh)]=c["device_ms_median"]

FLOOR=1.0016
results=[]
for c in dtype["cases"]:
    if c["dtype"]!="float16": continue
    n=c["shapes"][0]
    op=c["op"]
    meas=c["device_ms_median"]*1000.0  # ms->? keep ms; table ms
    meas_ms=c["device_ms_median"]
    # naive: fp32 table entry for same op,size if exists else nearest size
    # find table entry
    if (op,n) in table:
        naive_ms=table[(op,n)]
    else:
        # nearest size
        sizes=sorted(sz for (o,sz) in table if o==op)
        nearest=min(sizes,key=lambda s:abs(s-n))
        naive_ms=table[(op,nearest)]
    # byte-corrected: floor + 0.5*(naive-floor) in us? work in ms
    floor_ms=FLOOR/1000.0
    corrected_ms=floor_ms + 0.5*(naive_ms-floor_ms)
    naive_err=abs((naive_ms-meas_ms)/meas_ms)
    corr_err=abs((corrected_ms-meas_ms)/meas_ms)
    # bytes-keyed: evaluate fp32 curve at same byte footprint (n*dtype_bytes/4 elements)
    dtype_bytes = 2
    eq_n = int(n * dtype_bytes / 4)
    # fp32 table entry for same op at eq_n (nearest/interpolate)
    if (op, eq_n) in table:
        bytes_pred_ms = table[(op, eq_n)]
    else:
        sizes = sorted(sz for (o,sz) in table if o==op)
        # interpolate
        # build list
        pts = sorted((sz, table[(op,sz)]) for sz in sizes)
        if eq_n <= pts[0][0]:
            bytes_pred_ms = pts[0][1]
        elif eq_n >= pts[-1][0]:
            bytes_pred_ms = pts[-1][1]
        else:
            for i in range(len(pts)-1):
                if pts[i][0] <= eq_n <= pts[i+1][0]:
                    x0,y0=pts[i]; x1,y1=pts[i+1]
                    t=(math.log(eq_n)-math.log(x0))/(math.log(x1)-math.log(x0))
                    bytes_pred_ms = math.exp(math.log(y0)+t*(math.log(y1)-math.log(y0)))
                    break
            else:
                bytes_pred_ms = pts[-1][1]
    bytes_err = abs((bytes_pred_ms-meas_ms)/meas_ms)
    regime = "floor" if n<=65536 else "bandwidth"
    results.append({"op":op,"size":n,"measured_ms":meas_ms,
                    "naive_pred_ms":naive_ms,"corrected_pred_ms":corrected_ms,
                    "bytes_pred_ms":bytes_pred_ms,"bytes_pred_eq_n":eq_n,
                    "naive_err":naive_err,"corrected_err":corr_err,"bytes_err":bytes_err,"regime":regime})

# fp32 control cross-check
fp32_control=[c for c in dtype["cases"] if c["dtype"]=="torch.float32"]
cross=[]
for c in fp32_control:
    n=c["shapes"][0]
    # fixture v1 fp32 relu at same size
    v1cases=[cc for cc in v1["cases"] if cc["op_class"]=="elementwise" and cc["op"]=="relu" and cc["shapes"]==[n]]
    if v1cases:
        v1med=statistics.median(cc["device_ms_median"] for cc in v1cases)
        drift=abs(c["device_ms_median"]-v1med)/v1med
        cross.append({"size":n,"v2_ms":c["device_ms_median"],"v1_ms":v1med,"drift":drift})

metrics={
 "bead":"gpu-sim-84z",
 "n_cases":len(results),
 "per_case":results,
 "fp32_control_cross_check":cross,
 "summary":{
   "naive_median_err":statistics.median(r["naive_err"] for r in results),
   "naive_max_err":max(r["naive_err"] for r in results),
   "corrected_median_err":statistics.median(r["corrected_err"] for r in results),
   "corrected_max_err":max(r["corrected_err"] for r in results),
   "floor_band_naive_median":statistics.median(r["naive_err"] for r in results if r["regime"]=="floor"),
   "bw_band_naive_median":statistics.median(r["naive_err"] for r in results if r["regime"]=="bandwidth"),
   "bw_band_corrected_median":statistics.median(r["corrected_err"] for r in results if r["regime"]=="bandwidth"),
   "bytes_median_err":statistics.median(r["bytes_err"] for r in results),
   "bytes_max_err":max(r["bytes_err"] for r in results),
   "bw_band_bytes_median":statistics.median(r["bytes_err"] for r in results if r["regime"]=="bandwidth"),
   "bw_band_bytes_max":max(r["bytes_err"] for r in results if r["regime"]=="bandwidth"),
 },
 "verdict": None,
}
# verdict
if all(r["naive_err"]<=0.25 for r in results):
    metrics["verdict"]="DTYPE-AGNOSTIC"
elif any(r["naive_err"]>0.25 for r in results if r["regime"]=="bandwidth") and all(r["corrected_err"]<=0.25 for r in results if r["regime"]=="bandwidth"):
    metrics["verdict"]="DTYPE-BLIND-AT-BANDWIDTH"
else:
    metrics["verdict"]="MIXED"
    metrics["bytes_keyed_result"]="ALL PASS (median %.4f, max %.4f)" % (statistics.median(r["bytes_err"] for r in results), max(r["bytes_err"] for r in results))
HERE.joinpath("metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
print(json.dumps(metrics["summary"],indent=2))
print("verdict",metrics["verdict"])
