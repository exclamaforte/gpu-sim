#!/usr/bin/env bash
# gpu-sim-8y7.1 recompute: derive metrics.json from the committed sweep JSONs
# (pure CPU) and diff against the committed metrics.json. Never writes a
# tracked file: recompute goes to $SCRATCH, then cmp.
# Fails if any acceptance decomposition does not hold exactly.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/8y7.1-recompute-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
export CUDA_VISIBLE_DEVICES=
nice -n 10 python3 - "$HERE" "$SCRATCH/metrics.json" <<'EOF'
import json, sys
here, out = sys.argv[1], sys.argv[2]
cfgs = {}
for n in ['dp4', 'tp4', 'pp2', 'pp4']:
    cfgs[n] = json.load(open(f"{here}/sweep-{n}.json"))[0]

def row(n):
    r = cfgs[n]
    return {"step_ms": round(r["step_ms"], 1), "coll_msgs": r["coll_msgs"],
            "coll_MB": r["coll_MB"], "p2p_msgs": r["p2p_msgs"],
            "p2p_MB": r["p2p_MB"], "comm_sim_ms": r["comm_sim_ms"]}

checks = {
    # grank0 over 2 measured steps; exact decomposition, post-grank-fix only.
    "pp2_coll_is_18dp_x2_plus_barrier": cfgs['pp2']['coll_msgs'] == 18 * 2 + 1,
    "pp2_p2p_is_4per_step_x2": cfgs['pp2']['p2p_msgs'] == 4 * 2,
    "pp2_p2p_bytes_is_8_x_2.1MB": cfgs['pp2']['p2p_MB'] == round(8 * 2.097152, 1),
    "pp4_coll_is_barrier_only": cfgs['pp4']['coll_msgs'] == 1 and cfgs['pp4']['coll_MB'] == 0.0,
    "pp4_p2p_is_4per_step_x2": cfgs['pp4']['p2p_msgs'] == 4 * 2,
    "dp4_is_37params_x2_plus_barrier": cfgs['dp4']['coll_msgs'] == 37 * 2 + 1,
    "pp_bytes_below_tp_bytes": (cfgs['pp2']['coll_MB'] + cfgs['pp2']['p2p_MB']) < cfgs['tp4']['coll_MB'],
    "pp4_fewer_msgs_than_tp4": (cfgs['pp4']['coll_msgs'] + cfgs['pp4']['p2p_msgs']) < cfgs['tp4']['coll_msgs'],
}
m = {"bead": "gpu-sim-8y7.1",
     "configs": {n: row(n) for n in ['dp4', 'tp4', 'pp2', 'pp4']},
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
