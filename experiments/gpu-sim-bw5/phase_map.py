"""gpu-sim-bw5: parallelism phase map (CPU only)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from epic_cluster import parallelism_search as ps

MODELS = {
  "llama3-8b": {"n_layers":32,"d_model":4096,"n_heads":32,"head_dim":128,"d_ff":14336,"vocab":128256,"seq":8192},
  "llama3-70b": {"n_layers":80,"d_model":8192,"n_heads":64,"head_dim":128,"d_ff":28672,"vocab":128256,"seq":8192},
}
FABRICS = {
  "8xH100-NVLink": ("specs/h100_rack.json", 8),
  "16xH100-2node-IB": ("specs/h100_rack.json", 16),  # reuse 2-node spec but N=16 (spec has 16 ranks)
  "32xH100-4node-IB": ("specs/h100_rack_4node.json", 32),
}
# Note: 8-GPU fabric only has 8 ranks; N=16/32 on it impossible. Map N to fabric capacity.
N_FABRIC = {
  8: ["8xH100-NVLink"],
  16: ["8xH100-NVLink", "16xH100-2node-IB"],  # 8x can't do 16; but we run 16 on 2-node
  32: ["32xH100-4node-IB"],
}
# Actually restrict: N must <= fabric capacity
def fabrics_for_N(N):
    out=[]
    for name,(spec,cap) in FABRICS.items():
        if N <= cap:
            out.append(name)
    return out

records=[]
for N in (8,16,32):
    for fab in fabrics_for_N(N):
        spec_path = FABRICS[fab][0]
        for model_name, base_m in MODELS.items():
            for S in (8192, 32768):
                m = dict(base_m); m["seq"] = S
                hw = {"peak_flops":989e12,"mem_gb":80.0,"spec_path":spec_path}
                _, feas_sum = ps.search(N, hw, m=m, rank_by="sum")
                _, feas_max = ps.search(N, hw, m=m, rank_by="max")
                w_sum = feas_sum[0] if feas_sum else None
                w_max = feas_max[0] if feas_max else None
                # tie count at max floor
                if feas_max:
                    minstep = min(c.step_max for c in feas_max)
                    tied = sum(1 for c in feas_max if abs(c.step_max-minstep) < 1e-9)
                else:
                    tied = 0
                feasible = len(feas_sum) > 0
                if feasible:
                    c = w_sum
                    comm = c.tp_comm+c.cp_comm+c.pp_comm+c.dp_comm
                    constrained = "memory" if not feasible else ("comm" if c.dominant in ("tp","cp","pp","dp") else "compute")
                    rec = {
                        "N": N, "fabric": fab, "model": model_name, "seq": S,
                        "feasible": feasible,
                        "constrained_by": constrained,
                        "best_sum": {"tp":w_sum.tp,"pp":w_sum.pp,"cp":w_sum.cp,"dp":w_sum.dp,"zero":w_sum.zero,"step_ms":round(w_sum.step_sum*1000,2)} if w_sum else None,
                        "best_max": {"tp":w_max.tp,"pp":w_max.pp,"cp":w_max.cp,"dp":w_max.dp,"zero":w_max.zero,"step_ms":round(w_max.step_max*1000,2)} if w_max else None,
                        "tie_count_max_floor": tied,
                        "feasible_configs": len(feas_sum),
                        "total_configs": len(ps.search(N,hw,m=m)[0]),
                        "compute_ms": round(w_sum.compute*1000,2) if w_sum else None,
                        "comm_ms": round((w_sum.tp_comm+w_sum.cp_comm+w_sum.pp_comm+w_sum.dp_comm)*1000,2) if w_sum else None,
                        "mem_gb_winner": round(w_sum.memory_gb,1) if w_sum else None,
                        "mem_terms_gb_winner": {
                            "params": round(w_sum.mem_params_gb,2),
                            "grads": round(w_sum.mem_grads_gb,2),
                            "fp32_master": round(w_sum.mem_master_gb,2),
                            "opt_state": round(w_sum.mem_opt_gb,2),
                            "activations": round(w_sum.mem_act_gb,2),
                            "attn_workspace": round(w_sum.mem_attn_ws_gb,2),
                            "logits": round(w_sum.mem_logits_gb,2),
                            "cp_buffers": round(w_sum.mem_cp_buf_gb,2),
                        } if w_sum else None,
                    }
                else:
                    rec = {"N":N,"fabric":fab,"model":model_name,"seq":S,"feasible":False,"constrained_by":"memory",
                           "best_sum":None,"best_max":None,"tie_count_max_floor":0,"feasible_configs":0,
                           "total_configs":len(ps.search(N,hw,m=m)[0])}
                records.append(rec)
                print(f"N={N} {fab} {model_name} S={S}: feasible={feasible} winner_sum={rec['best_sum']} tied={rec['tie_count_max_floor']}")

with open("experiments/gpu-sim-bw5/metrics.json","w") as f:
    json.dump({"records": records}, f, indent=2)
print("written", len(records), "records")
