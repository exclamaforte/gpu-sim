#!/usr/bin/env bash
# gpu-sim-u2o.6.6.6.2 -- CPU only, run directly. Never through ts.
# The suite uses TEMPORARY queue directories and never touches the shared queue.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-u2o.6.6.6.2"
echo "=== sample provenance: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 - <<'PY' 2>&1 | tee "$ART/output.log"
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("qs", "epic_u2o/queue_state.py")
qs = importlib.util.module_from_spec(spec); spec.loader.exec_module(qs)
jobs = qs._load_all()
active = [j for j in jobs if j.get("state") in qs.ACTIVE]
if not active:
    print("NOTE: no active job to identify; cannot build an audit record "
          "(the contract is correct about this). Showing the loader's "
          "provenance directly."); print(json.dumps(qs._load_all_with_provenance()[1], indent=2))
else:
    st = qs.queue_state(own_job_id=active[0].get("id"))
    print(json.dumps(st["sample_provenance"], indent=2))
    print("total keys in record:", len(st))
PY
rc=${PIPESTATUS[0]}; echo "=== provenance exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"
echo "=== test suite ==="
python3 tests/test_sample_provenance.py 2>&1 | tee -a "$ART/output.log" | tail -4
rc=${PIPESTATUS[0]}; echo "=== tests exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"
echo "=== run.sh: all green ==="
