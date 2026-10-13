#!/bin/bash
# gpu-sim-u2o.12.1 replay: GPU numbers are documented, not replayed.
# Verifies the derivation (metrics == job output + ratio) and reruns the
# CPU preflight (shape/sync-guard path, tiny batch).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.12.1
EXP=$WT/experiments/gpu-sim-u2o.12.1
JOB=/home/gabe/Documents/gpu-sim/runs/gpu-sim-u2o.12.1/step_profile.json
cd "$WT" || exit 1
python3 - "$EXP/metrics.json" "$JOB" <<'EOF' || exit 1
import json, sys
m, j = (json.load(open(p)) for p in sys.argv[1:3])
for k in j:
    assert m[k] == j[k], f"drift in {k}"
assert abs(m["train_faster_than_gen_x"] - m["train_examples_per_sec"] / m["datagen_examples_per_sec"]) < 1e-9, "ratio direction"
assert m["model"]["n_params"] == 38950658, "v1 shape"
print("metrics == job output, ratio recomputes, v1 shape pinned")
EOF
CUDA_VISIBLE_DEVICES= python3 "$EXP/profile_step_121.py" --out /tmp/121replay --batch 2 --warmup 1 --steps 2 >/dev/null || exit 1
echo "REPLAY-OK"
