import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.torch_parallel.train import tp_groups, pp_neighbors

def layout_groups(pp, tp, cp, dp):
    # expected: for each (p, c, d), vary t
    out = []
    for p in range(pp):
        for c in range(cp):
            for d in range(dp):
                out.append([((p * tp + t) * cp + c) * dp + d for t in range(tp)])
    return out

CASES = [(2,1,1),(2,1,2),(2,2,1),(2,4,1),(4,2,1)]
def test_tp_groups():
    for tp, cp, dp in CASES:
        for pp in (1, 2):
            got = tp_groups(pp, tp, cp, dp)
            exp = layout_groups(pp, tp, cp, dp)
            assert got == exp, (pp, tp, cp, dp, got, exp)

def test_pp_neighbors():
    # pp=2, tp=2, cp=2, dp=2: stride = tp*cp*dp = 8
    prev, nxt = pp_neighbors(0, 2, 2, 2, 2)
    assert prev is None
    assert nxt == 8
    prev, nxt = pp_neighbors(8, 2, 2, 2, 2)
    assert prev == 0
    assert nxt is None
    # cp_rank preserved: rank 1 (cp_rank 1?) -> neighbour 9
    prev, nxt = pp_neighbors(1, 2, 2, 2, 2)
    assert nxt == 9

def test_validator_logic():
    # synthetic: equal counts -> valid
    import tempfile, json, os
    d = tempfile.mkdtemp()
    trace = os.path.join(d, "t.jsonl")
    summ = os.path.join(d, "s.jsonl")
    # two ranks, equal counts
    with open(summ, "w") as f:
        for r in (0,1):
            f.write(json.dumps({"rank": r, "measure_start_ns": 0}) + "\n")
    with open(trace, "w") as f:
        for r in (0,1):
            for _ in range(3):
                f.write(json.dumps({"grank": r, "ts_ns": 10, "op": "allreduce"}) + "\n")
    from tools.torch_parallel.validate_trace import counts_per_rank
    per = counts_per_rank(trace, summ)
    assert per[0]["allreduce"] == per[1]["allreduce"] == 3


def test_validator_detects_inequality():
    import tempfile, json, os, subprocess, sys
    d = tempfile.mkdtemp()
    trace = os.path.join(d, "t.jsonl")
    summ = os.path.join(d, "s.jsonl")
    with open(summ, "w") as f:
        for r in (0,1):
            f.write(json.dumps({"rank": r, "measure_start_ns": 0}) + "\n")
    with open(trace, "w") as f:
        # rank 0: 3 allreduce; rank 1: 1 allreduce (broken group)
        for _ in range(3):
            f.write(json.dumps({"grank": 0, "ts_ns": 10, "op": "allreduce"}) + "\n")
        f.write(json.dumps({"grank": 1, "ts_ns": 10, "op": "allreduce"}) + "\n")
    res = subprocess.run([sys.executable, "-m", "tools.torch_parallel.validate_trace",
                         "--trace", trace, "--summary", summ],
                        capture_output=True, text=True)
    assert res.returncode != 0, res.stdout + res.stderr
    assert "INVALID" in (res.stdout + res.stderr)
