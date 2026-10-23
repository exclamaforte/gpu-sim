#!/usr/bin/env bash
# gpu-sim-8y7.3 recompute: derive metrics.json from the committed sweep JSONs
# (pure CPU) and diff against the committed metrics.json. Never writes a
# tracked file: recompute goes to $SCRATCH, then cmp.
# Gates the acceptance decomposition exactly (counts) and relationally (bytes).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/8y7.3-recompute-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
export CUDA_VISIBLE_DEVICES=
nice -n 10 python3 - "$HERE" "$SCRATCH/metrics.json" <<'EOF'
import json, sys
here, out = sys.argv[1], sys.argv[2]
cfgs = {}
for n in ['zero1', 'zero2', 'zero3']:
    cfgs[n] = json.load(open(f"{here}/sweep-{n}.json"))[0]

def row(n):
    r = cfgs[n]
    return {"step_ms": round(r["step_ms"], 1), "coll_msgs": r["coll_msgs"],
            "coll_MB": r["coll_MB"], "comm_sim_ms": r["comm_sim_ms"],
            "peak_MB": r["peak_MB"], "mem_model_MB": r["mem_model_MB"]}

z1, z2, z3 = cfgs['zero1'], cfgs['zero2'], cfgs['zero3']
checks = {
    # 37 parameter tensors (incl. pos/LN the analytic count excludes), 2 steps.
    "zero1_is_75dp_plus_74gather_plus_barrier": z1['coll_msgs'] == (37 * 2 + 1) + 37 * 2,
    # DP part (43.0) + gather part (43.0) as separately rounded table values
    # sum to 86.0 against the rounded total 86.1 -- tolerance covers the
    # rounding; exactness lives in the message-count check above.
    "zero1_bytes_are_dp_plus_gather": abs(z1['coll_MB'] - (43.0 + 43.0)) < 0.15,
    # 1 reduce_scatter + 37 all_gathers per step, + barrier.
    "zero2_is_2rs_plus_74gather_plus_barrier": z2['coll_msgs'] == 2 + 37 * 2 + 1,
    # 1 allgather + 1 reduce_scatter per step (FSDP), + barrier.
    "zero3_is_2ag_plus_2rs_plus_barrier": z3['coll_msgs'] == 2 + 2 + 1,
    # The deliverable trade-off: per-rank traffic bytes fall with N...
    "bytes_fall_1_2_3": z1['coll_MB'] > z2['coll_MB'] > z3['coll_MB'],
    # ...and the extra collectives cost time from 1 to 2 (hand-written path).
    "step_rises_1_to_2": z2['step_ms'] > z1['step_ms'],
    # Analytic state model shards as specified (P=21.0MB analytic).
    "model_states_shard": (
        z1['mem_model_MB']['opt_states'] == z2['mem_model_MB']['opt_states'] and
        abs(z2['mem_model_MB']['grads'] * 4 - z1['mem_model_MB']['grads']) < 0.2 and
        abs(z3['mem_model_MB']['params'] * 4 - z1['mem_model_MB']['params']) < 0.2),
}
m = {"bead": "gpu-sim-8y7.3",
     "configs": {n: row(n) for n in ['zero1', 'zero2', 'zero3']},
     "note": ("measured peak_MB is activation-dominated and barely moves; "
              "the sharded-state savings live in mem_model_MB, reported alongside"),
     "decomposition_checks": checks,
     "all_checks_pass": all(checks.values())}
json.dump(m, open(out, "w"), indent=2, sort_keys=True)
open(out, "a").write("\n")
print(json.dumps(checks, indent=2))
assert all(checks.values()), "DECOMPOSITION CHECK FAILED"
print("all decomposition checks pass")
EOF
if cmp -s "$HERE/metrics.json" "$SCRATCH/metrics.json"; then
    echo "RECOMPUTE PASS: metrics match committed sweep JSONs"
else
    echo "RECOMPUTE FAIL: metrics differ"; diff "$HERE/metrics.json" "$SCRATCH/metrics.json" || true
    exit 1
fi
