#!/usr/bin/env bash
# gpu-sim-8dw -- recompute check. CPU-ONLY: no CUDA, no ts queue.
# Re-runs verify_8dw.py into scratch and compares the stable verdict fields
# against the committed metrics.json. Writes nothing tracked.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/8dw-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/verify_8dw.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
tail -n 4 "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: verifier exited $rc"; exit 1; }
python3 - "$HERE/metrics.json" "$SCRATCH/recomputed.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
f = json.load(open(sys.argv[2]))
ok = True
for k in ("verdict", "failures"):
    if c.get(k) != f.get(k):
        print(f"MISMATCH {k}: {c.get(k)!r} vs {f.get(k)!r}"); ok = False
for crit in ("C1", "C2", "C3", "C4"):
    if c["criteria"][crit] != f["criteria"][crit]:
        print(f"MISMATCH {crit}: {c['criteria'][crit]!r} vs {f['criteria'][crit]!r}")
        ok = False
if f["criteria"]["C5"]["code_hits"]:
    print(f"MISMATCH C5 code_hits: {f['criteria']['C5']['code_hits']!r}"); ok = False
print("RECOMPUTE PASS" if ok else "RECOMPUTE FAIL")
sys.exit(0 if ok else 1)
EOF
