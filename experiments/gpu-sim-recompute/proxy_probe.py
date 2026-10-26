"""Where does recomputation actually change the ANSWER? (proxy: act_factor x retained fraction)

Activation memory is proportional to act_factor, so scaling act_factor by the measured retained fraction
(0.33 at seq 8192, see README.md) is a faithful proxy for full checkpointing in the MEMORY term. It does not
charge the 1.25x recompute TIME cost, so the step times below are optimistic for the recompute rows.

Finding: the proxy does NOT change any 405B answer, at 80 or 800 GPUs -- because at pp=100/dp=1 the model
charges 14 B/param of optimizer state, which is 56.7 GB/rank, and activations are only ~19 GB. The knob's
real effect appears in batch-heavy regimes, where activations bind:

  70B, 80 GPUs, batch=8    no recompute (1,20,4,1,0) 0.749 s 65.9 GB   -> unchanged by recompute
  70B, 80 GPUs, batch=32   no recompute (1, 4,1,20,z3) 2.769 s 58.0 GB 1.6%
  70B, 80 GPUs, batch=32   recompute 0.33  (1, 1,1,80,z3) 2.474 s 70.9 GB 10.6%

That last row is the point: with recomputation the model stops buying pipeline stages to hold activations
and buys DP instead -- pp=1, pure FSDP over all 80 GPUs, 10.2% faster and far better separated. This is the
regime real frontier training lives in (large per-rank batch), not the optimizer-bound 405B cells.
"""
import importlib.util
import sys

sys.path.insert(0, ".")
_sm = importlib.util.spec_from_file_location("so", "scripts/scaleout_answers.py")
so = importlib.util.module_from_spec(_sm)
_sm.loader.exec_module(so)
from epic_cluster import parallelism_search as ps

MODELS = {
    "70b": {"n_layers": 80, "d_model": 8192, "n_heads": 64, "n_kv_heads": 8, "head_dim": 128,
            "d_ff": 28672, "vocab": 128256, "seq": 8192},
    "405b": {"n_layers": 126, "d_model": 16384, "n_heads": 128, "n_kv_heads": 8, "head_dim": 128,
             "d_ff": 53248, "vocab": 128256, "seq": 8192},
}


def answer(nodes, model, batch, act_mult):
    spec = so.build_spec(nodes)
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec": spec, "term_scale": {"cp": 1.9922},
          "act_factor": 2.3 * act_mult, "batch": batch}
    _, feas = ps.search(nodes * 8, hw, model)
    feas = [c for c in feas if c.tp <= 8 and c.pp <= model["n_layers"]]
    if not feas:
        return "INFEASIBLE"
    b = feas[0]
    alt = next(c for c in feas if (c.tp, c.pp, c.cp, c.dp) != (b.tp, b.pp, b.cp, b.dp))
    return {"config": f"tp={b.tp} pp={b.pp} cp={b.cp} dp={b.dp} zero={b.zero}",
            "step_s": round(b.step_sum, 3), "mem_gb": round(b.memory_gb, 1),
            "margin_pct": round(100 * (alt.step_sum / b.step_sum - 1), 2)}


if __name__ == "__main__":
    for model_name in ("70b", "405b"):
        for nodes in (10, 100):
            for batch in (8, 32):
                for label, mult in (("none", 1.0), ("recompute", 0.33)):
                    print(f"{model_name} {nodes * 8:4d} GPUs batch={batch:2d} {label:9s}:",
                          answer(nodes, MODELS[model_name], batch, mult), flush=True)
