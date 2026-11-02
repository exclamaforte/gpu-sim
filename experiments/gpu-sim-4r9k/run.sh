#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - << 'PY'
import json
RESIDUAL={1024:0.018,16384:0.068,262144:0.078,4194304:0.004,16777216:0.027}
def nr(n):
    ks=sorted(RESIDUAL); return RESIDUAL[min(ks,key=lambda k:abs(k-n))]
rows=[json.loads(l) for l in open("experiments/gpu-sim-4r9k/measured_gemm.jsonl")]
usable=0
for r in rows:
    n=r["size"]; residual=nr(n)
    threshold=5*residual if residual>0 else 5*max(r["within_std_us"],residual)
    gc = r["grid"][0]*r["grid"][1]*r["grid"][2] > 65535
    if r["kernel_us"]>=threshold and not gc and r["kernel_us"]>0:
        usable+=1
print(f"usable after fix: {usable}/{len(rows)}")
assert usable==24, f"expected 24, got {usable}"
print("OK")
PY
