"""gpu-sim-qk9: re-rank after DP comm audit."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from epic_cluster import parallelism_search as ps

MODELS = {"8b": {"n_layers":32,"d_model":4096,"n_heads":32,"head_dim":128,"d_ff":14336,"vocab":128256}}
SEQS = [8192]
NS = [8, 16, 32]
FABRIC = {8: "nvlink4_h100", 16: "specs/h100_rack.json", 32: "specs/h100_rack_4node.json"}

def hw(N, dp_dtype="bf16", dp_count_mode="flat", zero_semantics="production"):
    if N == 8:
        h = {"peak_flops":989e12,"mem_gb":80.0,"fabric":"nvlink4_h100","dp_dtype":dp_dtype,"dp_count_mode":dp_count_mode}
    else:
        h = {"peak_flops":989e12,"mem_gb":80.0,"spec_path":FABRIC[N],"dp_dtype":dp_dtype,"dp_count_mode":dp_count_mode}
    return h

# Compare old (fp32, zero>=2 does RS+AG) vs new (bf16, zero-level semantics)
# Old model: simulate by using dp_dtype=fp32 and old zero semantics. But we changed semantics.
# Instead, just report new FSDP row.
results = {}
for N in NS:
    m = dict(MODELS["8b"]); m["seq"] = 8192
    h = hw(N)
    _, feas = ps.search(N, h, m=m)
    if feas:
        w = feas[0]
        fsdp = [c for c in feas if c.tp==1 and c.pp==1 and c.cp==1 and c.dp==8 and c.zero==3]
        f = fsdp[0] if fsdp else None
        results[f"N{N}"] = {
            "winner": {"tp":w.tp,"pp":w.pp,"cp":w.cp,"dp":w.dp,"zero":w.zero,"step_ms":round(w.step_sum*1000,2),"dp_comm_ms":round(w.dp_comm*1000,3)},
            "fsdp_row": {"step_ms":round(f.step_sum*1000,2),"dp_comm_ms":round(f.dp_comm*1000,3)} if f else None,
        }
        print(f"N={N}: winner={w.tp},{w.pp},{w.cp},{w.dp},z={w.zero} step={w.step_sum*1000:.2f} dp_comm={w.dp_comm*1000:.3f}ms | FSDP step={f.step_sum*1000:.2f} dp_comm={f.dp_comm*1000:.3f}ms" if f else f"N={N}: no FSDP")

json.dump(results, open("experiments/gpu-sim-qk9/rerank.json","w"), indent=2)
print("written")
