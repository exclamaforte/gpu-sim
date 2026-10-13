#!/usr/bin/env bash
# gpu-sim-f6y RECOMPUTE (CPU only): replay every CPU-side step and require
# byte-identical outputs. The GPU job (measure_f6y.sh, job
# 20261011t145826-efc35eab) is NOT replayed: its outputs
# (measured_f6y_raw_stamped.jsonl, output.log, measure_summary.json walls)
# are records, and re-measuring would be new data, not a recompute.
# Exit nonzero on any gate failure.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=

echo "=== 0. oob-guard negative control + manifest sweep ==="
python3 -m pytest ../../tests/test_oob_guard.py -q -p no:cacheprovider || exit 1

echo "=== 1. build-manifest gate replays (guard over the 80) ==="
python3 - <<'PYEOF'
import json, sys
sys.path.insert(0, "../..")
from epic_u2o.kernel_driver import synthesize
man = json.load(open("manifest.json"))["specs"]
assert len(man) == 80, len(man)
for s in man:
    synthesize({k: s[k] for k in ("source", "kernel_name", "grid", "block",
                                  "args", "problem_size", "shared_mem_bytes",
                                  "op_type", "metadata")})
print("80/80 synthesize under current templates")
PYEOF
[ $? -eq 0 ] || exit 1

echo "=== 2. merge + census replay byte-identical ==="
sha_new_before=$(sha256sum measured_new.jsonl | cut -d" " -f1)
sha_cen_before=$(sha256sum census.json | cut -d" " -f1)
python3 merge_f6y.py || exit 1
python3 census_new.py || exit 1
sha_new_after=$(sha256sum measured_new.jsonl | cut -d" " -f1)
sha_cen_after=$(sha256sum census.json | cut -d" " -f1)
[ "$sha_new_before" = "$sha_new_after" ] || { echo "measured_new.jsonl NOT byte-identical"; exit 1; }
[ "$sha_cen_before" = "$sha_cen_after" ] || { echo "census.json NOT byte-identical"; exit 1; }
echo "byte-identical merge + census"

echo "=== 3. verify ==="
python3 verify.py || exit 1

echo "=== 4. registry shape ==="
python3 -m pytest ../../tests/test_registry_shape.py -q -p no:cacheprovider || exit 1

echo "=== RECOMPUTE PASS ==="
