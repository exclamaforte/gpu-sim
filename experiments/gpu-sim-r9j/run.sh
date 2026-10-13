#!/usr/bin/env bash
# gpu-sim-r9j RUN (CPU, unqueued): re-gather evidence + regression tests.
# Fails on mismatch: evidence.json must be byte-identical (same live-state
# modulo counts are NOT asserted -- instead the invariant keys are checked),
# and the test suite must exit 0.
set -euo pipefail
HERE="$(dirname "$0")"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=

rm -rf /tmp/r9jcheck; mkdir -p /tmp/r9jcheck
cp gather_evidence.py /tmp/r9jcheck/
python3 /tmp/r9jcheck/gather_evidence.py > /tmp/r9j_run_evidence.log 2>&1
tail -2 /tmp/r9j_run_evidence.log
python3 -c "
import json
fresh = json.load(open('/tmp/r9jcheck/evidence.json'))
committed = json.load(open('evidence.json'))
assert fresh['distinct_ids'] == committed['distinct_ids'] == ['lab-implementer', 'lab-pi', 'lab-reviewer']
assert fresh['one_file_per_cwd'] is True
assert fresh['global_lookup_exists'] is False
assert [s['site'] for s in fresh['resolution_sites']] == [s['site'] for s in committed['resolution_sites']]
ev = committed
assert ev['distinct_ids'] == ['lab-implementer', 'lab-pi', 'lab-reviewer'], ev['distinct_ids']
assert ev['one_file_per_cwd'] is True
assert ev['global_lookup_exists'] is False
assert all(s['scope'] == 'repo-scoped' for s in ev['resolution_sites']), ev['resolution_sites']
print('evidence invariants HOLD: 3 shared ids, 1-file-per-cwd, no global lookup')
"
cd ../..
python3 tests/test_session_scope.py 2>&1 | tail -4
echo "RUN PASS: r9j evidence + scope tests green"
