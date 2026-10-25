"""Pin the payload convention at BOTH ends of the interposer -> cost-model path.

WHY THIS EXISTS. gpu-sim-4x8 found the interposer feeding fg_nccl_predict the
PER-RANK send count for all-gather and reduce-scatter while link_model's ring
factors are defined on the TOTAL payload. The fix (pass count*nr) was verified by
a GPU trace, but the implementer's own negative control showed the fix is
INVISIBLE to the test suite: reverting it passes every existing test, because
those tests exercise the analytical nccl_cost path, not link_model. So there was
no regression guard, and the next edit could silently restore an nr-x
understatement in predictions, in trace bytes, and in the wall-clock step times
the harness sleeps.

Two guards, because the convention lives in two places:
  1. link_model's arithmetic: for a ring all-gather/reduce-scatter, the per-rank
     traffic is (n-1)/n of the TOTAL, so predict(total=4c) must cost 4x
     predict(total=c) at n=4 and must equal predict(reduce_scatter, 4c).
  2. The C call site: the predictor must be handed the total, which only the
     source can show without a GPU run.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster.link_model import predict  # noqa: E402

from tests.test_link_model import uniform_spec  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src", "fakegpu_nccl.c")
C = 4096  # per-rank chunk in elements (float32 -> 16 KiB)


def _t(alg, count, spec):
    return predict(spec, alg, count, "float32", pieces_per_phase=1)["duration_s"]


def test_allgather_costs_the_total_not_the_chunk():
    # With zero link latency the duration is pure bandwidth, so the two payloads
    # differ by exactly the group size -- that is the quantity the interposer used
    # to feed the predictor as a per-rank chunk instead of a total.
    spec0 = uniform_spec(4, lat=0.0)
    one = _t("ring_allgather", C, spec0)          # one rank's chunk
    four = _t("ring_allgather", 4 * C, spec0)     # the total those chunks make up
    assert four == pytest.approx(4 * one), (
        "ring_allgather must scale with the TOTAL payload: passing the per-rank "
        "chunk understates the prediction by the group size (gpu-sim-4x8)")

    # Two phases of the SAME total: equal bandwidth, so this holds exactly only
    # when latency is out of the picture (the latency factors differ by design --
    # an all-reduce pays two phase latencies, a one-phase ring pays one).
    assert _t("ring_allreduce", 4 * C, spec0) == pytest.approx(2 * four)

    # Whatever the latency does, the two one-phase ring algorithms must agree on
    # the same TOTAL payload.
    spec = uniform_spec(4)
    assert _t("ring_allgather", 4 * C, spec) == pytest.approx(
        _t("ring_reduce_scatter", 4 * C, spec))


def test_predictor_call_sites_pass_the_total():
    src = open(SRC).read()
    ag = re.search(r'fg_nccl_predict\(t, "ring_allgather", ([^,]+),', src)
    rs = re.search(r'fg_nccl_predict\(t, "ring_reduce_scatter", ([^,]+),', src)
    assert ag, "allgather predict call not found -- did the interposer change?"
    assert rs, "reduce_scatter predict call not found -- did the interposer change?"
    assert "nr" in ag.group(1), (
        f"allgather feeds the predictor {ag.group(1).strip()!r}; the total is needed "
        f"(count*nr), otherwise the prediction is nr x too small (gpu-sim-4x8)")
    assert "nr" in rs.group(1), (
        f"reduce_scatter feeds the predictor {rs.group(1).strip()!r}; the total is "
        f"needed (rc*nr) (gpu-sim-4x8)")

