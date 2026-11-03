#!/usr/bin/env bash
# gpu-sim-vb7x -- recompute check. CPU-ONLY: no CUDA, no ts queue.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
mkdir -p "${REPO}/.lab/tmp"
SCRATCH="$(mktemp -d "${REPO}/.lab/tmp/vb7x-recompute-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
{
  echo "=== pytest test_fs2y_mutations.py ==="
  CUDA_VISIBLE_DEVICES= python -m pytest tests/test_fs2y_mutations.py -q
  echo "pytest_rc=$?"
  echo "=== check_append_only.py ==="
  python scripts/check_append_only.py
  echo "check_rc=$?"
} > "$SCRATCH/output.log" 2>&1
cat "$SCRATCH/output.log"
# generate metrics into scratch
python3 - "$SCRATCH/metrics.json" <<'EOF'
import json, subprocess, sys
# count registry entries
import importlib.util
spec = importlib.util.spec_from_file_location("m", "scripts/fs2y_mutations.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
reg_size = len(mod.MUTATIONS)
import json as j
ledger = j.load(open("tests/fs2y_ledger.json"))
non_pending = sum(1 for e in ledger["entries"] if e.get("state") != "pending")
out = {"registry_size": reg_size, "ledger_non_pending": non_pending}
json.dump(out, open(sys.argv[1], "w"), indent=2)
open(sys.argv[1], "a").write("\n")
EOF
# compare
python3 - "$HERE/metrics.json" "$SCRATCH/metrics.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
f = json.load(open(sys.argv[2]))
if c != f:
    print("MISMATCH", c, f); sys.exit(1)
print("RECOMPUTE PASS")
EOF
rc=$?
exit $rc