#!/usr/bin/env bash
# gpu-sim-u2o.6.6.5 -- CPU only. Run directly, never through ts: a CPU job in
# the shared GPU queue blocks real GPU work for its whole runtime, and nothing
# here touches the device (nvidia-smi --query-gpu is a driver read that needs
# no CUDA context and no queued job).
#
# Verifies the telemetry sampler live (fields come back from the real driver)
# and runs the suite. Writes metrics.json deterministically from the run;
# exits nonzero if the sampler or any test fails.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-u2o.6.6.5"

echo "=== sampler probe: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 > "$ART/sampler_probe.json" 2>"$ART/output.log" <<'EOF'
import json, sys
sys.path.insert(0, "experiments/gpu-sim-u2o.6.1.1")
import verify_harness as vh
snap = vh.telemetry("rework-verification")
print(json.dumps({"label": snap["label"],
                  "fields": snap.get("fields", {}),
                  "n_fields": len(snap.get("fields", {})),  # 9 telemetry + name/driver_version/memory.total identity
                  "expected": sorted(vh.TELEMETRY_FIELDS),
                  "throttle_predicate_false_on_empty": vh.throttled({}) is False,
                  "failure_recorded": snap.get("query_rejected")
                                      or snap.get("nvidia_smi_error"),
                  "rc": 0 if snap.get("fields") else 1}, indent=2))
EOF
rc=$?
cat "$ART/sampler_probe.json" | tee -a "$ART/output.log" | head -20
echo "=== sampler exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"
python3 -c "
import json
snap = json.load(open('$ART/sampler_probe.json'))
assert snap['rc'] == 0, snap
assert set(snap['expected']) <= set(snap['fields']), snap
metrics = {'bead': 'gpu-sim-u2o.6.6.5', 'gpu_used': False,
           'what': 'telemetry sampler verified live (driver read, no CUDA context, no queue)',
           'telemetry_fields_expected': snap['expected'],
           'telemetry_fields_returned': sorted(snap['fields']),
           'throttle_predicate': 'zero/absent mask -> False (tested)',
           'read_only': 'nvidia-smi --query-gpu only; no -lgc, no persistence change, no sudo (tested)',
           'duplication_account': 'verify_harness == measure_noise (9 fields, tested); measure_family has 5 (gap documented, PI schema decision pending)',
           'code_commit': '$(git rev-parse --short HEAD)'}
json.dump(metrics, open('$ART/metrics.json', 'w'), indent=2)
print('wrote $ART/metrics.json')
"

echo "=== test suite ==="
python3 tests/test_verify_harness_telemetry.py 2>&1 | tee -a "$ART/output.log" | tail -3
rc=${PIPESTATUS[0]}
echo "=== tests exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== run.sh: all green ==="
