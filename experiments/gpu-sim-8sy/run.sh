#!/usr/bin/env bash
# gpu-sim-8sy: correct m3d.3 n=2 factors (bytes + latency) and verify.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="$(pwd)/experiments/gpu-sim-8sy"
mkdir -p "$OUT"
LOG="$OUT/output.log"
: > "$LOG"

echo "=== make check ===" | tee -a "$LOG"
make check >> "$LOG" 2>&1

echo "=== pytest test_link_model.py ===" | tee -a "$LOG"
python3 -m pytest tests/test_link_model.py -q >> "$LOG" 2>&1

echo "=== m3d.7 run.sh (C/Python agreement) ===" | tee -a "$LOG"
bash experiments/gpu-sim-m3d.7/run.sh >> "$LOG" 2>&1

echo "=== public-route factor table ===" | tee -a "$LOG"
python3 - <<'PY' >> "$LOG" 2>&1
import sys
sys.path.insert(0, ".")
from epic_cluster.link_model import predict
for n in (2, 3, 4, 8):
    spec = {"version":1,"models":{"m":{"sm_count":80,"clock_mhz":1000,"mem_bytes":1<<30,"mem_bw_gbs":1000.0,"arch":"x","source":"test"}},
            "nodes":[{"name":"n0","devices":[{"id":i,"model":"m"} for i in range(n)]}],
            "links":[{"a":{"node":"n0","device":i},"b":{"node":"n0","device":i+1},"kind":"nvlink","bw_gbps":100.0,"latency_us":5.0,"mtu":4096,"loss_ppm":0,"jitter_us":0.0,"shared_group":None} for i in range(n-1)],
            "ranks":[{"rank":i,"node":"n0","device":i} for i in range(n)]}
    r = predict(spec, "ring_allreduce", 1024, "float32", n_ranks=n)
    per_rank = (2.0 if n == 2 else 2.0*(n-1)/n) * 4096
    print(n, "per_rank_bytes", per_rank, "duration_s", r["duration_s"])
PY

echo "=== DONE ===" | tee -a "$LOG"
