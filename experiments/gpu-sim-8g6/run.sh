#!/usr/bin/env bash
# gpu-sim-8g6 -- recompute check. CPU-ONLY: the three GPU probe jobs cannot be
# recomputed without a device. Re-runs build_verdict.py into scratch and
# byte-compares against the committed metrics.json (modulo the recorded tree
# commit, which is ancestor-checked). Writes nothing tracked.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/u8g6-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/build_verdict.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: verifier exited $rc"; exit 1; }
python3 - "$HERE/metrics.json" "$SCRATCH/recomputed.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
f = json.load(open(sys.argv[2]))
for k in ("verdict_final", "failures"):
    assert c.get(k) == f.get(k), f"MISMATCH {k}"
cc, ff = dict(c), dict(f)
cc.pop("tree", None); ff.pop("tree", None)
assert json.dumps(cc, sort_keys=True) == json.dumps(ff, sort_keys=True), "MISMATCH payload"
print("payload match modulo recorded tree")
EOF
[ "$?" -eq 0 ] || exit 1
TREE_COMMIT=$(python3 - "$HERE/metrics.json" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1]))["tree"]["commit"])
EOF
)
git -C "$REPO" merge-base --is-ancestor "$TREE_COMMIT" HEAD \
    || { echo "RECOMPUTE FAIL: recorded tree $TREE_COMMIT not an ancestor"; exit 1; }
echo "tree $TREE_COMMIT verified; RECOMPUTE PASS"
