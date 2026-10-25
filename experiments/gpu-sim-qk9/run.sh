#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 experiments/gpu-sim-qk9/rerank.py
# consistency check: artifact winner step == search() winner step
python3 - <<'PY'
import sys, json, os
sys.path.insert(0, ".")
from epic_cluster import parallelism_search as ps
d = json.load(open("experiments/gpu-sim-qk9/rerank.json"))
for key, v in d.items():
    N = int(key[1:])
    m = {"n_layers":32,"d_model":4096,"n_heads":32,"head_dim":128,"d_ff":14336,"vocab":128256,"seq":8192}
    if N <= 16:
        hw = {"peak_flops":989e12,"mem_gb":80.0,"spec_path":"specs/h100_rack.json","dp_dtype":"bf16"}
    else:
        hw = {"peak_flops":989e12,"mem_gb":80.0,"spec_path":"specs/h100_rack_4node.json","dp_dtype":"bf16"}
    _, feas = ps.search(N, hw, m=m)
    assert feas, f"N={N} no feasible"
    w = feas[0]
    assert abs(v["winner"]["step_ms"] - round(w.step_sum*1000, 2)) < 1e-6, \
        f"N={N}: artifact {v['winner']['step_ms']} != search {round(w.step_sum*1000,2)}"
print("consistency OK")
PY
