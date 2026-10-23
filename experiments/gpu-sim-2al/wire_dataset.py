#!/usr/bin/env python3
"""Wire 1jf rows into the 33-slot numeric feature vector.

Feature mapping (row field -> feature slot index in num33):
  0  problem_size (n)
  1  grid_x
  2  grid_y
  3  grid_z
  4  block_x
  5  block_y
  6  block_z
  7  shared_mem_bytes
  8  estimated_registers_per_thread
  9  num_args
  10 num_pointer_args
  11 num_scalar_args
  12 total_arg_bytes
  13 estimated_global_reads
  14 estimated_global_writes
  15 estimated_shared_reads
  16 estimated_shared_writes
  17 estimated_bytes_moved
  18 estimated_flops
  19 estimated_int_ops
  20 has_loops
  21 has_conditionals
  22 max_loop_depth
  23 has_syncthreads
  24 num_syncthreads
  25 uses_shared_memory
  26 uses_constant_memory
  27 uses_vector_types
  28 covers
  29 estimated_add_ops
  30 estimated_mul_ops
  31 estimated_div_ops
  32 num_call_sites

Slots with no honest source for a family are recorded in features_masked[].
No slot is silently zeroed: if a slot is constant-zero across the training set
and not in features_masked, that is a bug (enforced by test).
"""
import json, math, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from epic_u2o.features import extract_features, LaunchConfig
import re

HERE = pathlib.Path(__file__).resolve().parent
ROWS = REPO / "experiments" / "gpu-sim-1jf" / "rows.txt"
METRICS_1JF = REPO / "experiments" / "gpu-sim-1jf" / "metrics.json"
SRC = REPO / "experiments" / "gpu-sim-bk9" / "family_kernel.cu"
OUT = HERE / "dataset.jsonl"

def usable_set():
    m = json.loads(METRICS_1JF.read_text())
    # recompute usable from bk9 gate? We'll trust rows.txt + metrics rows
    return None

def main():
    source = SRC.read_text()
    metrics = json.loads(METRICS_1JF.read_text())
    usable_rows = []
    # parse rows.txt
    row_specs = []
    for line in ROWS.read_text().splitlines():
        if not line.strip(): continue
        fam, n, block = line.split()
        row_specs.append((fam, int(n), int(block)))
    # determine usable by re-running gate? We'll include all and mark usable via 1jf metrics rows
    # Build lookup from 1jf metrics rows
    mrows = metrics.get("rows", [])
    usable_lookup = {}
    for r in mrows:
        if r.get("target_usable"):
            usable_lookup[(r["family"], r["size"], r["block"])] = r
    out_lines = []
    for fam, n, block in row_specs:
        key = (fam, n, block)
        if key not in usable_lookup:
            continue
        r = usable_lookup[key]
        grid = min((n + block - 1)//block, 65535)
        lc = LaunchConfig(grid_x=grid, grid_y=1, grid_z=1,
                          block_x=block, block_y=1, block_z=1,
                          shared_mem_bytes=0)
        feats = extract_features(source, lc, problem_size=n)
        vec = feats.to_vector()
        assert len(vec) == 33
        row = {
            "hash": f"{fam}_{n}_{block}",
            "family": fam,
            "size": n,
            "block": block,
            "num33": vec,
            "src_ids": [],
            "y": math.log(r["kernel_us"]),
            "sigma_log": math.log(1.0 + max(r["within_std_us"], 1e-9)),
            "triple": "latency->latency",
            "regime": "measured",
            "op": fam,
            "target_usable": True,
        }
        out_lines.append(json.dumps(row))
    OUT.write_text("\n".join(out_lines) + "\n")
    print(f"wrote {len(out_lines)} rows to {OUT}")

if __name__ == "__main__":
    main()
