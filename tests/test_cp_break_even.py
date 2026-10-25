"""gpu-sim-54m: regression test for CP break-even sequence length."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

def s_best(N=8, path="legacy", model=None, cp_grad_mode="crude", act_factor=2.3):
    if model is None:
        model = {"n_layers":32,"d_model":4096,"n_heads":32,"head_dim":128,"d_ff":14336,"vocab":128256}
    seqs = [8192,16384,32768,65536,131072]
    for seq in seqs:
        m = dict(model); m["seq"] = seq
        if N == 8 and path == "legacy":
            hw = {"peak_flops":989e12,"mem_gb":80.0,"fabric":"nvlink4_h100","cp_grad_mode":cp_grad_mode,"act_factor":act_factor}
        elif N == 8:
            hw = {"peak_flops":989e12,"mem_gb":80.0,"spec_path":"specs/h100_rack.json","cp_grad_mode":cp_grad_mode,"act_factor":act_factor}
        elif N <= 16:
            hw = {"peak_flops":989e12,"mem_gb":80.0,"spec_path":"specs/h100_rack.json","cp_grad_mode":cp_grad_mode,"act_factor":act_factor}
        else:
            hw = {"peak_flops":989e12,"mem_gb":80.0,"spec_path":"specs/h100_rack_4node.json","cp_grad_mode":cp_grad_mode,"act_factor":act_factor}
        _, feas = ps.search(N, hw, m=m)
        if feas and feas[0].cp > 1:
            return seq
    return None

def test_s_best_legacy_n8_8b():
    # Legacy single-fabric path: FSDP survives? No -- FSDP infeasible (85 GB > 80)
    # but cp_grad on nccl path makes cp=2 slower than cp=1 at 16384, so S_best=32768.
    assert s_best(path="legacy") == 32768, s_best(path="legacy")

def test_s_best_spec_n8_8b():
    # REVISED after the DP-comm audit (gpu-sim-qk9). The original expectation here was
    # S_best=16384 on the spec path (cp=2 at 200.09 vs cp=1 at 201.98 ms). bf16 DP
    # payloads + real ZeRO semantics moved the cp=1 rows, and the spec path now agrees
    # with legacy: cp=1 wins at 16384 and S_best is 32768. Production cp-grad still
    # pulls it in to 16384 -- asserted separately so the knob keeps its meaning.
    assert s_best(path="spec") == 32768, s_best(path="spec")
    assert s_best(path="spec", cp_grad_mode="production") == 16384, \
        s_best(path="spec", cp_grad_mode="production")
    assert s_best(path="legacy") == 32768, s_best(path="legacy")

def test_act_factor_override_changes_memory():
    # hw.get('act_factor', ...) must actually affect the memory number.
    m = {"n_layers":32,"d_model":4096,"n_heads":32,"head_dim":128,"d_ff":14336,"vocab":128256,"seq":16384}
    hw1 = {"peak_flops":989e12,"mem_gb":80.0,"fabric":"nvlink4_h100","act_factor":2.3}
    hw2 = {"peak_flops":989e12,"mem_gb":80.0,"fabric":"nvlink4_h100","act_factor":1.0}
    c1 = ps.score(1,1,1,8,3,hw1,m)
    c2 = ps.score(1,1,1,8,3,hw2,m)
    assert c1.memory_gb != c2.memory_gb, (c1.memory_gb, c2.memory_gb)
    assert c1.memory_gb > c2.memory_gb
