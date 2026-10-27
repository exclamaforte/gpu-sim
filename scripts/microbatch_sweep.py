#!/usr/bin/env python3
"""Micro-batch sensitivity of the N=8 / N=16 verdict (gpu-sim-oggn).

Sweeps micro-batches per stage (the factor micro_batches/pp) over 1,2,4,8,16,32 and
records the winner for the N=8 and N=16 cells. Pure simulator; no GPU.

Usage: python3 scripts/microbatch_sweep.py [--json]
"""
from __future__ import annotations
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epic_cluster import parallelism_search as ps

PEAK = 989e12
SPECS = {8: "specs/h100_rack.json", 16: "specs/h100_rack.json"}
MODEL = {"n_layers": 32, "d_model": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128,
         "d_ff": 14336, "vocab": 128256}
FACTORS = [1, 2, 4, 8, 16, 32]


def hw(n, factor):
    return {"peak_flops": PEAK, "mem_gb": 80.0, "spec_path": SPECS[n], "seq": 8192,
            "micro_batches_mult": factor}


def main() -> int:
    out = {}
    for n in (8, 16):
        rows = []
        prev = None
        trans = 0
        first = None
        for factor in FACTORS:
            h = hw(n, factor)
            m = dict(MODEL, seq=8192)
            _, feas = ps.search(n, h, m)
            if not feas:
                shape = None
                step = None
                total_mb = None
            else:
                w = feas[0]
                shape = [w.tp, w.pp, w.cp, w.dp, w.zero]
                step = round(w.step_sum * 1e3, 2)
                total_mb = w.pp * factor
            is_trans = (prev is not None and shape != prev)
            if is_trans:
                trans += 1
                if first is None:
                    first = factor
            rows.append({"factor": factor, "total_micro_batches": total_mb,
                         "verdict": shape, "step_ms": step, "transition": is_trans})
            prev = shape
        out[f"N{n}"] = {"rows": rows, "transitions": trans,
                        "first_transition_factor": first}
    if "--json" in sys.argv:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    for key, v in out.items():
        print(f"{key}: transitions={v['transitions']} first_factor={v['first_transition_factor']}")
        for r in v["rows"]:
            print(f"  factor={r['factor']:2d} total_mb={r['total_micro_batches']:3d} verdict={r['verdict']} step={r['step_ms']}{' <--' if r['transition'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
