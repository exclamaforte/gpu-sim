"""gpu-sim-oggn: pin the micro-batch transition counts and first-transition factors."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import microbatch_sweep as ms

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "experiments" / "gpu-sim-oggn" / "metrics.json").read_text())


def test_n8_transition_count():
    assert METRICS["N8"]["transitions"] == 1


def test_n8_first_transition_factor():
    assert METRICS["N8"]["first_transition_factor"] == 2


def test_n16_transition_count():
    assert METRICS["N16"]["transitions"] == 1


def test_n16_first_transition_factor():
    assert METRICS["N16"]["first_transition_factor"] == 16


def test_base_verdict_n8():
    assert METRICS["N8"]["rows"][0]["verdict"] == [4, 1, 2, 1, 0]


def test_sweep_reproduces():
    out = {}
    for n in (8, 16):
        rows = []
        prev = None
        trans = 0
        first = None
        for factor in ms.FACTORS:
            h = ms.hw(n, factor)
            m = dict(ms.MODEL, seq=8192)
            _, feas = __import__("epic_cluster.parallelism_search", fromlist=["x"]).search(n, h, m)
            shape = [feas[0].tp, feas[0].pp, feas[0].cp, feas[0].dp, feas[0].zero] if feas else None
            is_trans = prev is not None and shape != prev
            if is_trans:
                trans += 1
                if first is None:
                    first = factor
            rows.append({"factor": factor, "verdict": shape})
            prev = shape
        assert trans == METRICS[f"N{n}"]["transitions"]
        assert first == METRICS[f"N{n}"]["first_transition_factor"]
