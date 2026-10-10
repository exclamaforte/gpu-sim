#!/usr/bin/env bash
# gpu-sim-u2o.19 -- recompute check. Rebuilds libfakegpu + reruns the overlay
# gate CPU-side, re-parses the two recorded ts probe logs, byte-compares
# metrics. Writes no tracked file (gate log + recomputed metrics to scratch).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
SCRATCH="$(mktemp -d /tmp/u19-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/build"
CUDA_VISIBLE_DEVICES= nice -n 10 make -j4 >/dev/null 2>&1
CUDA_VISIBLE_DEVICES= nice -n 10 bash -c "/home/gabe/.local/bin/nvcc -cudart=none -arch=sm_120 -Xlinker -rpath -Xlinker $REPO tests/test_device_prop_overlay.cu -o $SCRATCH/build/ov -L$REPO -lfakegpu && $SCRATCH/build/ov" > "$SCRATCH/gate.log" 2>&1
tail -n 3 "$SCRATCH/gate.log"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/u19_verify.py" \
    --metrics-out "$SCRATCH/recomputed.json" --gate-log "$SCRATCH/gate.log" 2>&1 | tail -n 1
rc=$?
[ "$rc" -eq 0 ] || [ "$rc" -eq 1 ] || { echo "RECOMPUTE FAIL: crashed ($rc)"; exit 1; }
cmp -s "$HERE/metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical, verdict rc=$rc)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
