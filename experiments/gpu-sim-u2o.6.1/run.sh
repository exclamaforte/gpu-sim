#!/usr/bin/env bash
# gpu-sim-u2o.6.1 -- recompute check. CPU-ONLY: no CUDA, no ts queue.
# Re-runs tests/test_process_isolation_timeout.py into a scratch dir and
# compares the stable verdict fields against the committed metrics.json.
# Writes nothing tracked; leaves `git status` clean. Fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/u2o61-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$REPO/tests/test_process_isolation_timeout.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: test exited $rc"; exit 1; }
python3 - "$HERE/metrics.json" "$SCRATCH/recomputed.json" <<'EOF'
import json, sys
committed = json.load(open(sys.argv[1]))
fresh = json.load(open(sys.argv[2]))
KEYS = ["bead", "timeout_sec", "verdict", "failures", "pipeline_continues"]
SUB = {"normal": ["success", "returncode", "timeout"],
       "hang": ["timeout", "success", "returncode", "survivors"]}
ok = True
for k in KEYS:
    if committed.get(k) != fresh.get(k):
        print(f"MISMATCH {k}: committed={committed.get(k)!r} fresh={fresh.get(k)!r}")
        ok = False
for sec, fields in SUB.items():
    for f in fields:
        if committed[sec].get(f) != fresh[sec].get(f):
            print(f"MISMATCH {sec}.{f}: committed={committed[sec].get(f)!r} "
                  f"fresh={fresh[sec].get(f)!r}")
            ok = False
# Fresh wall times must still sit inside the budget (volatile values, bound-checked).
t, w = fresh["timeout_sec"], fresh["hang"]["wall_time_sec"]
if not (t <= w <= t + 15):
    print(f"MISMATCH hang.wall_time_sec={w} outside [{t}, {t + 15}]")
    ok = False
print("RECOMPUTE PASS" if ok else "RECOMPUTE FAIL")
sys.exit(0 if ok else 1)
EOF
