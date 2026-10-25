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
FABRIC = {8: ("nvlink4_h100", None), 16: (None, "specs/h100_rack.json"), 32: (None, "specs/h100_rack_4node.json")}

def hw_for(N, cp_grad_mode="crude", act_factor=2.3):
    fabric, spec = FABRIC[N]
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "cp_grad_mode": cp_grad_mode, "act_factor_override": act_factor}
    if spec:
        hw["spec_path"] = spec
    else:
        hw["fabric"] = fabric
    return hw

def run_one(N, model_name, seq, rank_by, cp_grad_mode, act_factor):
    base = dict(MODELS[model_name]); base["seq"] = seq
    # act_factor override: monkeypatch by setting hw key; score() reads hw.get("act_factor"...)? 
    # Currently act_factor is hardcoded 2.3. To support 1.0, we need to pass it. 
    # We'll handle by temporarily scaling: use a wrapper. Simpler: set hw["act_factor"] and patch score.
    m = base
    hw = hw_for(N, cp_grad_mode, act_factor)
    hw["act_factor"] = act_factor
    _, feas = ps.search(N, hw, m=m, rank_by=rank_by)
    # Note: score() ignores hw["act_factor"] currently. We'll patch below.
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

# Patch score() to honor hw["act_factor"] if present
orig_score = ps.score
def patched_score(tp, pp, cp, dp, zero, hw, m):
    # temporarily set act_factor by monkeypatching the module global? 
    # Instead, we replicate: call orig, but orig uses hardcoded 2.3.
    # We'll just call orig; act_factor sensitivity handled by post-hoc? 
    # For simplicity, accept that act_factor is fixed at 2.3 in this run;
    # the sensitivity run will be a separate invocation with a patched module.
    return orig_score(tp, pp, cp, dp, zero, hw, m)
ps.score = patched_score

# To support act_factor sensitivity, we monkeypatch the source string
import re

def set_act_factor(val):
    src = open("epic_cluster/parallelism_search.py").read()
    # replace the act_factor line for "none"
    src2 = src.replace('        act_factor = 2.3   # calibrated', f'        act_factor = {val}   # calibrated')
    open("epic_cluster/parallelism_search.py","w").write(src2)

def restore_act_factor():
    src = open("epic_cluster/parallelism_search.py").read()
    src2 = src.replace('        act_factor = 1.0   # calibrated', '        act_factor = 2.3   # calibrated')
    src2 = src2.replace('        act_factor = 2.3   # calibrated', '        act_factor = 2.3   # calibrated')
    open("epic_cluster/parallelism_search.py","w").write(src2)

# Run grid
results = {}
for cp_grad_mode in ("crude", "production"):
    for act_factor in (2.3, 1.0):
        key = f"cpgrad={cp_grad_mode}_act={act_factor}"
        set_act_factor(act_factor)
        # reload module
        import importlib; importlib.reload(ps)
        grid = {}
        for N in NS:
            for model_name in ("8b", "70b"):
                for seq in SEQS:
                    for rb in ("sum", "max"):
                        cell = run_one(N, model_name, seq, rb, cp_grad_mode, act_factor)
                        grid[f"N{N}_{model_name}_S{seq}_{rb}"] = cell
        results[key] = grid
        print(f"done {key}")

# restore
set_act_factor(2.3)
import importlib; importlib.reload(ps)

# Compute thresholds
thresholds = {}
for cp_grad_mode in ("crude","production"):
    for act_factor in (2.3,1.0):
        key = f"cpgrad={cp_grad_mode}_act={act_factor}"
        grid = results[key]
        for N in NS:
            for model_name in ("8b","70b"):
                # S_best: smallest seq where best config has cp>1
                s_best = None
                s_req = None
                for seq in SEQS:
                    cell = grid[f"N{N}_{model_name}_S{seq}_sum"]
                    if not cell["feasible"]:
                        continue
                    if s_best is None and cell["best"]["cp"] > 1:
                        s_best = seq
                    if s_req is None and not cell["cp1_feasible"]:
                        s_req = seq
                thresholds[f"{key}_N{N}_{model_name}"] = {"S_best": s_best, "S_required": s_req}

json.dump({"results": results, "thresholds": thresholds}, open("experiments/gpu-sim-54m/sweep.json","w"), indent=2)
print("written sweep.json")
# print thresholds
for k,v in sorted(thresholds.items()):
    print(k, v)
