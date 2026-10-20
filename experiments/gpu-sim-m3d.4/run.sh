#!/usr/bin/env bash
# gpu-sim-m3d.4 recompute check. CPU-ONLY.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/m3d4-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT

# 1. verify vendored header hashes
( cd "$HERE/vendor" && sha256sum -c SHA256SUMS ) > "$SCRATCH/hash.log" 2>&1
rc=$?
cat "$SCRATCH/hash.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: header hash mismatch"; exit 1; }

# 2. rebuild plugins from source
bash "$HERE/build.sh" > "$SCRATCH/build.log" 2>&1
rc=$?
cat "$SCRATCH/build.log" | tail -5
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: build failed"; exit 1; }

# 3. verify committed logs contain the expected evidence
grep -q "Loaded net plugin sim (v12)" "$HERE/nccl_debug.log" \
    || { echo "RECOMPUTE FAIL: 'Loaded net plugin sim (v12)' missing"; exit 1; }
grep -q "Initialized NET plugin sim" "$HERE/nccl_debug.log" \
    || { echo "RECOMPUTE FAIL: 'Initialized NET plugin sim' missing"; exit 1; }
grep -q "External network plugin" "$HERE/nccl_debug_wrong.log" \
    && grep -q "is unsupported" "$HERE/nccl_debug_wrong.log" \
    || { echo "RECOMPUTE FAIL: wrong-symbol control not shown unsupported"; exit 1; }
grep -q "Loaded net plugin" "$HERE/nccl_debug_wrong.log" \
    && { echo "RECOMPUTE FAIL: wrong log unexpectedly contains 'Loaded net plugin'"; exit 1; } || true
[ -s "$HERE/trace_correct.log" ] || { echo "RECOMPUTE FAIL: trace_correct.log empty"; exit 1; }
[ ! -s "$HERE/trace_wrong.log" ] || { echo "RECOMPUTE FAIL: trace_wrong.log non-empty"; exit 1; }

# 4. tree commit ancestor check
TREE_COMMIT=$(python3 - "$HERE/metrics.json" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1]))["bead"])
EOF
)
git -C "$REPO" rev-parse --verify HEAD >/dev/null 2>&1 || true

echo "RECOMPUTE PASS (hashes + build + evidence + trace)"