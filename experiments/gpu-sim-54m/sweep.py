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

def hw_for(N, path="legacy", cp_grad_mode="crude", act_factor=None):
    if N == 8 and path == "legacy":
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100", "cp_grad_mode": cp_grad_mode}
    elif N == 8:
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": "specs/h100_rack.json", "cp_grad_mode": cp_grad_mode}
    elif N <= 16:
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": "specs/h100_rack.json", "cp_grad_mode": cp_grad_mode}
    else:
        hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": "specs/h100_rack_4node.json", "cp_grad_mode": cp_grad_mode}
    if act_factor is not None:
        hw["act_factor"] = act_factor
    return hw

def run_one(N, model_name, seq, rank_by, cp_grad_mode, act_factor, path="legacy"):
    m = dict(MODELS[model_name]); m["seq"] = seq
    hw = hw_for(N, path, cp_grad_mode, act_factor)
    _, feas = ps.search(N, hw, m=m, rank_by=rank_by)
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

# Run grid
summary = {}
for path in ("legacy", "spec"):
    for cp_grad_mode in ("crude", "production"):
        for act_factor in (2.3, 1.0):
            key = f"path={path}_cpgrad={cp_grad_mode}_act={act_factor}"
            thr = {}
            for N in NS:
                for model_name in ("8b", "70b"):
                    s_best = None; s_req = None
                    cells = {}
                    for seq in SEQS:
                        cell = run_one(N, model_name, seq, "sum", cp_grad_mode, act_factor, path)
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

# REVISED after the DP-comm audit (gpu-sim-qk9, 2026-10-25). The original assertion
# here encoded 54m's headline: at 16384 under default knobs (act=2.3, crude) the legacy
# path picks a cp=1 config while the spec path picks cp>1, so S_best was path-dependent
# (16384 spec vs 32768 legacy). After bf16 DP payloads + real ZeRO semantics the two
# paths AGREE: both pick cp=1 at 16384 and both put S_best at 32768. The remaining
# driver is the cp gradient-sync mode, not the pricing path -- production pulls N=8
# in to 16384. Assert the CURRENT fact, so the next model change breaks this line
# again instead of silently rewriting the doc.
leg = summary["path=legacy_cpgrad=crude_act=2.3"]["N8_8b"]["cells"][16384]["best"]
spec = summary["path=spec_cpgrad=crude_act=2.3"]["N8_8b"]["cells"][16384]["best"]
# post-6r5: cp>1 wins at 16384 on BOTH paths (the dp-heavy rows the earlier
# assertions were written against are no longer competitive).
assert leg["cp"] > 1, f"legacy cp at 16384 should be >1 after 6r5, got {leg['cp']}"
assert spec["cp"] > 1, f"spec cp at 16384 should be >1 after 6r5, got {spec['cp']}"
sb = {k: v["N8_8b"]["S_best"] for k, v in summary.items() if "N8_8b" in v}
# REVISED again after gpu-sim-6r5 (payload convention + ZeRO-1/2 param all-gather):
# the DP term was dp x too cheap, dp-heavy rows looked best, and CP appeared to lose at
# 8192. Corrected, cp>1 wins from the smallest sequence tested on both paths and under
# every knob combination -> S_best = 8192 everywhere.
assert sb["path=legacy_cpgrad=crude_act=2.3"] == 8192, sb
assert sb["path=spec_cpgrad=crude_act=2.3"] == 8192, sb
assert sb["path=spec_cpgrad=production_act=2.3"] == 8192, sb
print("ASSERT OK (post-6r5): S_best = 8192 on both paths, crude and production")

# Save compact
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
for key in sorted(compact):
    for nk in ("N8_8b",):
        v = compact[key][nk]
        print(f"{key} {nk}: S_best={v['S_best']} S_required={v['S_required']}")
