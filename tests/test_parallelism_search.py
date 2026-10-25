import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster.parallelism_search import search, MODEL

def test_n1_selects_all_one():
    hw = {"peak_flops": 1e12, "mem_gb": 200.0, "fabric": "nvlink4_h100"}
    results, feasible = search(1, hw)
    assert feasible
    w = feasible[0]
    assert (w.tp, w.pp, w.cp, w.dp) == (1, 1, 1, 1)

def test_increasing_n_not_increase_step_time():
    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100"}
    prev = None
    for N in [1, 2, 4, 8]:
        _, feas = search(N, hw)
        if not feas:
            continue
        best = feas[0].step_sum
        if prev is not None:
            assert best <= prev * 1.01, (N, best, prev)
        prev = best

def test_too_large_model_has_feasible_at_n8():
    # gpu-sim-6y1 completed the memory model (fp32 master weights, logits, attention
    # workspace, CP buffers). An 8B model now costs ~16 B/param before activations
    # (bf16 params+grads, fp32 master, Adam m+v), so 8 x 16 GB is genuinely
    # infeasible -- the old assertion encoded the model that omitted those terms.
    # The intent (a model too large for one GPU still fits at N=8) is kept, at the
    # memory level where it is actually true.
    small = {"peak_flops": 1e12, "mem_gb": 16.0, "fabric": "nvlink4_h100"}
    results, feas = search(8, small)
    assert not feas, [(c.tp, c.pp, c.cp, c.dp, c.memory_gb) for c in feas]
    assert all(c.reason.startswith("mem ") for c in results)
    hw = {"peak_flops": 1e12, "mem_gb": 24.0, "fabric": "nvlink4_h100"}
    _, feas2 = search(8, hw)
    assert len(feas2) > 0
