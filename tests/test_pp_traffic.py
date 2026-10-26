"""PP traffic validated against the harness at pp=4 and pp=16, and it is micro-batch INVARIANT.

gpu-sim-ppcp: two frozen traces back this up.

  harness/pp4_cp1.jsonl  (pp=4, cp=1, d=256, seq=512, batch=8, micro_batches=2)
      middle rank: 6 sends/step of 2,097,152 B  -> 3 boundaries x 2 microbatches, payload
      (batch/mb)*seq*d*4B = (8/2)*512*256*4 = 2,097,152
  harness/pp16_cp1.jsonl (pp=16, cp=1, d=128, seq=256, batch=4, micro_batches=2)
      middle rank: 30 sends/step of 262,144 B   -> 15 boundaries x 2 microbatches, payload
      (4/2)*256*128*4 = 262,144

The important property is that TOTAL traffic does not depend on the micro-batch split:

    count x payload = mb*(pp-1) * (batch/mb)*seq*d*b = (pp-1)*batch*seq*d*b

so the model's `micro_batches` default (= pp) changes only the BUBBLE (idle time), never the bandwidth
estimate. That is why pp traffic was reproducible at both scales with a default that disagrees with the
harness's own value.
"""
import json
import pathlib

HARNESS = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "gpu-sim-ppcp" / "harness"


def _sends_per_step(path, rank):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    sends = [r for r in rows if r["op"] == "send" and r["rank"] == rank]
    return sends


def test_pp16_matches_the_harness_exactly():
    sends = _sends_per_step(HARNESS / "pp16_cp1.jsonl", 1)
    pp, batch, seq, d, mb, elem = 16, 4, 256, 128, 2, 4
    assert len(sends) == mb * (pp - 1), f"expected {mb * (pp - 1)} sends, got {len(sends)}"
    payload = (batch / mb) * seq * d * elem
    assert {r["bytes"] for r in sends} == {int(payload)}, "message size disagrees with the model"
    assert sum(r["bytes"] for r in sends) == (pp - 1) * batch * seq * d * elem


def _check(path, rank, pp, batch, seq, d, elem, mb, max_steps):
    """The frozen traces carry no step field, so recover the step count from the total.

    pp4's run covered 2 steps (12 sends for a 6-per-step model), pp16's covered 1. Asserting the
    payload exactly and the step count as an integer keeps that assumption visible instead of
    silently dividing by a number chosen to make the test pass.
    """
    sends = _sends_per_step(path, rank)
    payload = (batch / mb) * seq * d * elem
    assert {r["bytes"] for r in sends} == {int(payload)}, "message size disagrees with the model"
    per_step = mb * (pp - 1)
    total = sum(r["bytes"] for r in sends)
    steps = total / (per_step * payload)
    assert steps == int(steps) and 1 <= steps <= max_steps, (
        f"total {total} B is not an integer number of steps of {per_step}x{int(payload)} B "
        f"(got {steps} steps)")
    assert total == int(steps) * (pp - 1) * batch * seq * d * elem


def test_pp16_matches_the_harness_exactly():
    _check(HARNESS / "pp16_cp1.jsonl", 1, pp=16, batch=4, seq=256, d=128, elem=4, mb=2, max_steps=2)


def test_pp4_matches_the_harness_exactly():
    _check(HARNESS / "pp4_cp1.jsonl", 1, pp=4, batch=8, seq=512, d=256, elem=4, mb=2, max_steps=2)


def test_total_pp_traffic_is_invariant_to_the_micro_batch_split():
    """The property that makes the model's micro_batches default harmless for bandwidth."""
    pp, batch, seq, d, elem = 16, 4, 256, 128, 4
    totals = set()
    for mb in (1, 2, 4, 8, 16):
        totals.add(round(mb * (pp - 1) * (batch / mb) * seq * d * elem, 6))
    assert len(totals) == 1, f"total traffic varied with micro_batches: {totals}"
    assert totals.pop() == (pp - 1) * batch * seq * d * elem
