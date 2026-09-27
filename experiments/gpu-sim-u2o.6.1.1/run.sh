#!/usr/bin/env bash
# gpu-sim-u2o.6.1.1 -- end-to-end harness verification on the real RTX 5090.
# Run ONLY through the shared ts queue (gpu-sim-u2o.6.8).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"

echo "=== ts job: $(hostname) at $(date -Is) ==="
command -v nvcc && nvcc --version | tail -2
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

cd "$HERE"
python3 verify_harness.py 2>&1 | tee output.log
rc=${PIPESTATUS[0]}
echo "=== verify_harness exit=$rc ==="
exit "$rc"
