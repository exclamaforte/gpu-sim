#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export U5R4_OUT="${U5R4_OUT:-$ROOT/.lab/tmp/5uu9_out}"
case "$U5R4_OUT" in
    "$ROOT"/.lab/tmp/*) ;;
    *) echo "REFUSE: U5R4_OUT must be under $ROOT/.lab/tmp (got '$U5R4_OUT'); refusing to rm -rf it" >&2; exit 2 ;;
esac
rm -rf "$U5R4_OUT"; mkdir -p "$U5R4_OUT"
python3 "$HERE/analyze.py"
python3 - "$U5R4_OUT/metrics.json" "$HERE/metrics.json" <<'PY'
import json,sys
def norm(o):
    o=json.loads(json.dumps(o))
    for k in ("loadavg_1_5_15","foreign_gpu_job"):
        o.pop(k,None)
    if isinstance(o.get("environment"),dict):
        for k in ("loadavg_1_5_15","foreign_gpu_job"):
            o["environment"].pop(k,None)
        if not o["environment"]: del o["environment"]
    return o
a=norm(json.load(open(sys.argv[1]))); b=norm(json.load(open(sys.argv[2])))
if a!=b:
    print("METRICS_MISMATCH"); print(json.dumps(a,sort_keys=True)[:500]); print("---"); print(json.dumps(b,sort_keys=True)[:500]); sys.exit(1)
print("METRICS_MATCH")
PY
