"""gpu-sim-54m: CP break-even in sequence length."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from epic_cluster import parallelism_search as ps

MODELS = {
  "8b": {"n_layers":32,"d_model":4096,"n_heads":32,"head_dim":128,"d_ff":14336,"vocab":128256},
  "70b": {"n_layers":80,"d_model":8192,"n_heads":64,"head_dim":128,"d_ff":28672,"vocab":128256},
}
SEQS = [8192, 16384, 32768, 65536, 131072]
NS = [8, 16, 32]
FABRIC = {8: (None, None), 16: (None, "specs/h100_rack.json"), 32: (None, "specs/h100_rack_4node.json")}

def hw_for(N, cp_grad_mode="crude"):
    if N == 8:
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "cp_grad_mode": cp_grad_mode}
    else:
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": FABRIC[N][1], "cp_grad_mode": cp_grad_mode}
    return hw

def make_score(act_factor):
    orig = ps.score
    def patched(tp, pp, cp, dp, zero, hw, m):
        r = orig(tp, pp, cp, dp, zero, hw, m)
        # scale act_bytes by act_factor/2.3 (model uses 2.3 internally)
        if hasattr(r, "mem_act_gb"):
            r.mem_act_gb = r.mem_act_gb * (act_factor / 2.3)
            r.memory_gb = r.memory_gb + r.mem_act_gb * (act_factor/2.3 - 1)
        return r
    return patched

def run_one(N, model_name, seq, rank_by, cp_grad_mode, act_factor):
    base = dict(MODELS[model_name]); base["seq"] = seq
    m = base
    hw = hw_for(N, cp_grad_mode)
    ps.score = make_score(act_factor)
    _, feas = ps.search(N, hw, m=m, rank_by=rank_by)
    ps.score = ps.score.__wrapped__ if hasattr(ps.score, "__wrapped__") else ps.score
    # restore original
    import importlib; importlib.reload(ps)
    if not feas:
        return {"feasible": False, "feasible_configs": 0, "cp1_feasible": False,
                "best": None, "top5": []}
    w = feas[0]
    cp1 = [c for c in feas if c.cp == 1]
    return {
        "feasible": True,
        "feasible_configs": len(feas),
        "cp1_feasible": len(cp1) > 0,
        "best": {"tp":w.tp,"pp":w.pp,"cp":w.cp,"dp":w.dp,"zero":w.zero,
                 "step_ms":round(w.step_sum*1000,2),"mem_gb":round(w.memory_gb,1)},
        "top5": [{"tp":c.tp,"pp":c.pp,"cp":c.cp,"dp":c.dp,"zero":c.zero,
                  "step_ms":round(c.step_sum*1000,2)} for c in feas[:5]],
    }

# Run grid, compact summary only
summary = {}
thresholds = {}
for cp_grad_mode in ("crude", "production"):
    for act_factor in (2.3, 1.0):
        key = f"cpgrad={cp_grad_mode}_act={act_factor}"
        # store only thresholds, not full grid
        thr = {}
        for N in NS:
            for model_name in ("8b", "70b"):
                s_best = None; s_req = None
                cells = {}
                for seq in SEQS:
                    cell = run_one(N, model_name, seq, "sum", cp_grad_mode, act_factor)
                    cells[seq] = {"feasible": cell["feasible"], "best": cell["best"],
                                 "cp1_feasible": cell["cp1_feasible"],
                                 "nfeas": cell["feasible_configs"]}
                    if cell["feasible"]:
                        if s_best is None and cell["best"]["cp"] > 1:
                            s_best = seq
                        if s_req is None and not cell["cp1_feasible"]:
                            s_req = seq
                thr[f"N{N}_{model_name}"] = {"S_best": s_best, "S_required": s_req, "cells": cells}
        summary[key] = thr
        print(f"done {key}")

# Save compact (thresholds + per-cell best, not top5)
compact = {}
for key, thr in summary.items():
    compact[key] = {}
    for nk, v in thr.items():
        compact[key][nk] = {"S_best": v["S_best"], "S_required": v["S_required"],
                            "cells": {str(s): {"feasible": c["feasible"], "best": c["best"],
                                              "cp1_feasible": c["cp1_feasible"],
                                              "nfeas": c["nfeas"]} for s,c in v["cells"].items()}}

json.dump({"thresholds_summary": compact}, open("experiments/gpu-sim-54m/sweep.json","w"), indent=2)
print("written compact sweep.json")
# print thresholds
for key in sorted(compact):
    for nk in ("N8_8b","N16_8b","N32_8b"):
        v = compact[key][nk]
        print(f"{key} {nk}: S_best={v['S_best']} S_required={v['S_required']}")
