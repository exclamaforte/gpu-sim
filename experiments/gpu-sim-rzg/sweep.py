"""gpu-sim-rzg: model-size sweep -- does the 8B answer generalize, and where does 80 GB bind?

CPU only. All numbers are MODELLED by epic_cluster.parallelism_search (spec/topology
pricing path). The 8B row is the audited baseline (gpu-sim-qk9): a sweep that does not
reproduce it is not trustworthy.

Model dicts are the published configs: layer count, d_model, n_heads, head_dim, d_ff,
vocab. seq fixed at 8192 (the regime the current answer is about).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from epic_cluster import parallelism_search as ps  # noqa: E402

MODELS = {
    "llama3-8b": {"n_layers": 32, "d_model": 4096, "n_heads": 32, "head_dim": 128,
                  "d_ff": 14336, "vocab": 128256},
    "qwen2.5-32b": {"n_layers": 64, "d_model": 5120, "n_heads": 40, "head_dim": 128,
                    "d_ff": 27648, "vocab": 152064},
    "llama2-34b": {"n_layers": 48, "d_model": 8192, "n_heads": 64, "head_dim": 128,
                   "d_ff": 22016, "vocab": 32000},
    "llama3-70b": {"n_layers": 80, "d_model": 8192, "n_heads": 64, "head_dim": 128,
                   "d_ff": 28672, "vocab": 128256},
}
SPEC = {"N8": "specs/h100_rack.json", "N16": "specs/h100_rack.json"}
N = {"N8": 8, "N16": 16}
SEQ = 8192


def main():
    out = {}
    for name, mdl in MODELS.items():
        m = dict(mdl)
        m["seq"] = SEQ
        rows = {}
        for key, spec_path in SPEC.items():
            hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": spec_path}
            _, feas = ps.search(N[key], hw, m)
            # minimum memory across the whole space, feasible or not -- the margin
            allcfg = []
            for tp in (1, 2, 4, 8):
                for pp in (1, 2, 4, 8):
                    for cp in (1, 2, 4, 8):
                        for dp in (1, 2, 4, 8, 16, 32):
                            if tp * pp * cp * dp != N[key]:
                                continue
                            for zero in (0, 1, 2, 3):
                                try:
                                    allcfg.append(ps.score(tp, pp, cp, dp, zero, hw, m))
                                except ValueError:
                                    pass
            minm = min(c.memory_gb for c in allcfg)
            argmin = min(allcfg, key=lambda c: c.memory_gb)
            row = {
                "seq": SEQ,
                "n_ranks": N[key],
                "spec": spec_path,
                "feasible": len(feas),
                "total": len(allcfg),
                "min_mem_gb": round(minm, 2),
                "min_mem_config": [argmin.tp, argmin.pp, argmin.cp, argmin.dp, argmin.zero],
            }
            if feas:
                b = feas[0]
                row["best"] = {
                    "config": [b.tp, b.pp, b.cp, b.dp, b.zero],
                    "step_ms": round(b.step_sum * 1e3, 2),
                    "mem_gb": round(b.memory_gb, 2),
                }
            rows[key] = row
            if feas:
                print(f"{name:12s} {key:3s} FIT   best={tuple(row['best']['config'])} "
                      f"step={row['best']['step_ms']:8.2f}ms mem={row['best']['mem_gb']:6.2f}GB "
                      f"({len(feas)}/{len(allcfg)} feasible)")
            else:
                print(f"{name:12s} {key:3s} MISS  min mem {minm:6.2f} GB at {tuple(row['min_mem_config'])}"
                      f" -- {'fits' if minm <= 80 else f'{minm - 80:.2f} GB over'}")
        out[name] = rows

    path = os.path.join(HERE, "sweep.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"written {path}")

    # Consistency: the 8B N8 row must reproduce the audited gpu-sim-qk9 answer.
    b = out["llama3-8b"]["N8"]["best"]
    assert tuple(b["config"]) == (1, 1, 1, 8, 2), f"8B baseline drifted: {b['config']}"
    assert abs(b["step_ms"] - 66.99) < 0.05, f"8B baseline step drifted: {b['step_ms']}"
    print("baseline check OK: llama3-8b N8 == 66.99 ms, zero=2 dp=8 (audited qk9)")


if __name__ == "__main__":
    main()
