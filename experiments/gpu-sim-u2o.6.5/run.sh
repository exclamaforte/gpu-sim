#!/usr/bin/env bash
# gpu-sim-u2o.6.5 -- recompute check. CPU half only: the GPU measurement
# (measured.json, ts job 20261006t010919-80b3abca) cannot be recomputed without
# a device, so run.sh re-verifies everything around it instead.
#  1. pair_results.py rerun to scratch must byte-match metrics.json;
#  2. build_corpus.py rerun to scratch must reproduce the verdict structure
#     (cids, ok flags, messages -- volatile cpu_sec excluded);
#  3. pairing assertions re-checked (measured==passed, rejects kept out).
# Writes nothing tracked. Fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/u65-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
export CUDA_VISIBLE_DEVICES=
echo "--- pair recompute ---"
nice -n 10 python3 "$HERE/pair_results.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/pair.log" 2>&1
rc=$?
cat "$SCRATCH/pair.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: pair exited $rc"; exit 1; }
python3 - "$HERE/metrics.json" "$SCRATCH/recomputed.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
f = json.load(open(sys.argv[2]))
c["submitted"].pop("tree", None)
f["submitted"].pop("tree", None)
if json.dumps(c, sort_keys=True) != json.dumps(f, sort_keys=True):
    print("RECOMPUTE FAIL: metrics differ (modulo tree)")
    sys.exit(1)
print("metrics match (modulo recorded tree)")
EOF
[ "$?" -eq 0 ] || exit 1
TREE_COMMIT=$(python3 - "$HERE/metrics.json" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1]))["submitted"]["tree"]["commit"])
EOF
)
git -C "$REPO" merge-base --is-ancestor "$TREE_COMMIT" HEAD \
    || { echo "RECOMPUTE FAIL: recorded tree $TREE_COMMIT not an ancestor"; exit 1; }
echo "tree $TREE_COMMIT verified as recorded measurement tree"
echo "--- corpus recompute (stable fields) ---"
nice -n 10 python3 "$HERE/build_corpus.py" \
    --out "$SCRATCH/corpus.json" > "$SCRATCH/corpus.log" 2>&1
rc=$?
tail -n 1 "$SCRATCH/corpus.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: corpus exited $rc"; exit 1; }
python3 - "$HERE/preflight_results.json" "$SCRATCH/corpus.json" <<'EOF'
import json, re, sys
c = json.load(open(sys.argv[1]))
f = json.load(open(sys.argv[2]))
ok = True
for k in ("n_corpus", "n_pass", "n_reject", "surprises", "verdict"):
    if c.get(k) != f.get(k):
        print(f"MISMATCH {k}: {c.get(k)!r} vs {f.get(k)!r}"); ok = False
norm = lambda m: re.sub(r"/tmp/[A-Za-z0-9._-]+", "/tmp/XXX", m)
cc = {p["cid"]: (p["ok"], norm(p["msg"])) for p in c["passes"] + c["rejects"]}
ff = {p["cid"]: (p["ok"], norm(p["msg"])) for p in f["passes"] + f["rejects"]}
if cc != ff:
    print("MISMATCH per-cid verdicts/messages"); ok = False
print("RECOMPUTE PASS" if ok else "RECOMPUTE FAIL")
sys.exit(0 if ok else 1)
EOF
