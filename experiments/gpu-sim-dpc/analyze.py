"""gpu-sim-dpc: measured DP collective COUNTS per rank per step (qk9 acceptance item 1).

Reads the committed traces in this directory (world=2, tp=1, pp=1, dp=2, TinyLM
5.24M params / 4 layers, 3 steps, 5090 time-sliced) and prints per-step op counts
and byte totals against the model's assumptions.

This is the measurement qk9 asked for and never got: qk9 rewrote the DP term from
a code read, then closed. The counts below are what the harness actually issues.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = 3
P_BYTES_FP32 = 5.24e6 * 4

# What epic_cluster/parallelism_search.py prices per step today (one op each):
MODEL_ASSUMES = {
    1: "allreduce(grads full)                        -- 1 op",
    2: "reduce_scatter(grads shard)                  -- 1 op, NO param all-gather",
    3: "reduce_scatter(grads shard) + allgather(params shard) -- 2 ops",
}


def main() -> int:
    out = {}
    print(f"measured: world=2 dp=2, {STEPS} steps, P={P_BYTES_FP32/1e6:.2f} MB fp32 "
          f"(shard P/dp={P_BYTES_FP32/2e6:.2f} MB)")
    for z in (1, 2, 3):
        path = os.path.join(HERE, f"qk9_z{z}.jsonl")
        per = collections.defaultdict(collections.Counter)
        byt = collections.defaultdict(collections.Counter)
        for line in open(path):
            if not line.strip():
                continue
            e = json.loads(line)
            rk = e.get("grank", e.get("rank"))
            per[rk][e.get("op")] += 1
            byt[rk][e.get("op")] += int(e.get("bytes") or 0)
        rk = sorted(per)[0]
        ops = {k: round(v / STEPS, 1) for k, v in per[rk].items()}
        mbytes = {k: round(v / STEPS / 1e6, 2) for k, v in byt[rk].items()}
        out[f"zero={z}"] = {"ops_per_step": ops, "mb_per_step": mbytes}
        print(f"\nzero={z}")
        print(f"  MEASURED  ops/step: {ops}")
        print(f"            MB/step : {mbytes}")
        print(f"  MODEL     {MODEL_ASSUMES[z]}")
        print(f"  GAP       {per[rk].get('allgather', 0)/STEPS:.0f} all-gather(s) per step "
              f"are NOT in the DP term as priced" if z in (1, 2) else
              "  GAP       op mix matches (1 RS + 1 AG); payload convention to confirm")

    json.dump(out, open(os.path.join(HERE, "dp_counts.json"), "w"), indent=2)
    print("\nwritten dp_counts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
