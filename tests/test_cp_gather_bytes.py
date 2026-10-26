"""The harness's CP collective moves ~2x the model's per-message bytes (gpu-sim, measured).

gpu-sim-cpgqa: with GQA support in the harness (--n-kv-heads), the CP all_gather is finally measurable
at the width the model prices. Three shapes, frozen under experiments/gpu-sim-cpgqa/:

    seq  kv_dim  model msg (seq/cp*kv*4)  harness msg   ratio
    512   256     262,144                 522,240       1.992
    256   256     131,072                 260,096       1.984
    512  1024     524,288               2,088,960       3.984  (2 gathers)

The last row shows the harness emitting TWO gathers for kv_dim=1024 (K and V separated once the KV width
exceeds the head count), so its per-tensor ratio stays ~2.0.

Why it matters: ~2.0x is OUTSIDE the N=8 verdict's robustness radius (1.5566x, gpu-sim-zj9a). The model
prices a RING exchange of shard-sized K/V chunks; the harness (like an all_gather-based CP implementation)
gathers the FULL sequence K/V on every rank. If the deployed stack uses the all_gather convention, the
model understates CP by the factor that overturns the recommendation.

Scope, stated honestly: this is BYTES, not durations, and it compares two legitimate algorithms -- the
doc already records that the CP axis is un-reconcilable by construction. What is new is that the
un-reconcilable gap now has a measured size, and that size is bigger than the error budget.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACES = ROOT / "experiments" / "gpu-sim-cpgqa"

# (trace, seq, kv_dim, expected number of K/V gathers per rank per step)
CASES = [("cp2gqa.jsonl", 512, 256), ("cp2s256.jsonl", 256, 256), ("cp2gqa8.jsonl", 512, 1024)]


def _kv_gather_size(path, n_layers=2):
    """The K/V gather happens once per layer per tensor (K and V), so it is the all_gather size that
    occurs exactly 2*n_layers times. Selecting by occurrence count is what distinguishes it from the
    activation-sized gathers (1x) and the parameter gathers (2x for two matrices)."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    counts = {}
    for r in rows:
        if r["op"] == "allgather" and r["rank"] == 0:
            counts[r["bytes"]] = counts.get(r["bytes"], 0) + 1
    hits = [size for size, n in counts.items() if n == 2 * n_layers]
    assert len(hits) == 1, f"{path.name}: expected exactly one size occurring {2 * n_layers}x, got {hits}"
    return hits[0]


def test_harness_cp_message_is_twice_the_model_message():
    """The measured convention gap: an all_gather CP moves ~2x what the model's ring exchange prices."""
    for name, seq, kv in CASES:
        size = _kv_gather_size(TRACES / name)
        model_msg = (seq // 2) * kv * 4          # seq/cp with cp=2
        ratio = size / model_msg
        assert 1.9 <= ratio <= 2.1, (
            f"{name}: harness CP gather {size} B is {ratio:.3f}x the model's {model_msg} B; "
            f"the measured convention gap is ~2.0x and the N=8 radius is 1.5566x")


def test_the_gap_exceeds_the_robustness_radius():
    """The whole point: 2.0x is bigger than the error budget that would overturn the N=8 verdict."""
    facts = json.loads((ROOT / "docs" / "model_facts.json").read_text())
    radius = facts["sensitivity_radius"]["N8_seq8192"]["flip_cp"]
    size = _kv_gather_size(TRACES / "cp2gqa.jsonl")
    gap = size / ((512 // 2) * 256 * 4)
    assert gap > radius, f"gap {gap:.2f}x is inside the radius {radius}: update the doc claim"
