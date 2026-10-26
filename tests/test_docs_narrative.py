"""Do the SENTENCES match the model, or only the decimals? (gpu-sim-narr)

`check_docs_model_numbers.py` proves every decimal in section 0 is a value the model can produce.
That is weaker than it looks: when the CP payload was corrected to the GQA K/V width, prose quoting
"6.42% at the implied MFU" for the N=8 margin kept passing, because 6.42 was still produced elsewhere
(it is now the CP advantage at 40% MFU). The numbers were individually legal and the sentences were
wrong. Three paragraphs shipped with stale claims behind a green decimal check.

This test closes that gap for the load-bearing sentences: it recomputes each value from the model and
requires the prose to carry THAT value. It is deliberately narrow -- a handful of claims that carry the
recommendation -- because a general "prose is true" test does not exist and pretending otherwise would
be the same mistake one level up.
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from epic_cluster import parallelism_search as ps

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "training-parallelism-sim.md").read_text()
FACTS = json.loads((ROOT / "docs" / "model_facts.json").read_text())


def _para(anchor):
    """The paragraph containing anchor (blank-line delimited)."""
    i = DOC.index(anchor)
    start = DOC.rfind("\n\n", 0, i) + 2
    end = DOC.find("\n\n", i)
    return DOC[start:end if end > 0 else len(DOC)]


def test_n8_margin_sentence_carries_current_margins():
    grid = FACTS["fragility_map"]["grid"]["N8_seq8192"]
    para = _para("The margins are what matter")
    for tag in ("implied", "measured", "typical"):
        val = f"{grid[tag]['margin_pct']:.2f}%"
        assert val in para, f"N=8 margin sentence does not carry {tag} margin {val}"


def test_n32_margin_sentence_carries_current_margins():
    grid = FACTS["fragility_map"]["grid"]["N32_seq8192"]
    para = _para("The margins are what matter")
    for tag in ("implied", "measured", "typical"):
        val = f"{grid[tag]['margin_pct']:.2f}%"
        assert val in para, f"N=32 margin sentence does not carry {tag} margin {val}"


def test_n16_sentence_carries_the_current_verdict_not_the_old_tie():
    """The N=16 claim INVERTED when the PP payload was priced correctly.

    gp·-8pwl: before the fix the prose said "N=16 is not a verdict at any efficiency" and the winner moved
    with MFU. After it, the winner is stable across all three efficiencies and the register calls it a
    VERDICT. The old test asserted the old property, which is why it kept passing while the prose and the
    facts disagreed. Now the test recomputes BOTH facts it depends on and requires the prose to state
    them: the winner must be identical at all three efficiencies, and only then may the prose claim
    stability.
    """
    grid = FACTS["fragility_map"]["grid"]
    cells = [grid[f"N16_seq{seq}"] for seq in (8192,)]
    winners = {tuple(cell[tag]["config"]) for cell in cells for tag in ("implied", "measured", "typical")}
    assert len(winners) == 1, f"N=16 winner is not stable across efficiency: {winners}"
    para = _para("N=16 became a verdict")
    for tag in ("implied", "measured", "typical"):
        val = f"{grid['N16_seq8192'][tag]['margin_pct']:.2f}%"
        assert val in para, f"N=16 paragraph does not carry the {tag} margin {val}"
    assert "same config" in para.lower(), "the paragraph must say the winner does not move with efficiency"


def test_prose_does_not_revive_the_old_n16_tie_claim():
    """Negative control: the sentence this repo used to publish must be gone."""
    assert "not a verdict at any efficiency" not in DOC


def test_n8_decisiveness_is_stated_per_sequence_length():
    """N=8 is decisive at 8192/32768 and a tie at 16384 -- the prose used to average that away."""
    grid = FACTS["fragility_map"]["grid"]
    tie = grid["N8_seq16384"]["typical"]["margin_pct"]
    assert tie < 2.0, f"N=8 seq16384 is no longer a tie ({tie}%), rewrite this test"
    para = _para("Decisiveness depends on sequence length")
    assert f"{tie:.2f}%" in para, f"the paragraph does not carry the seq-16384 tie margin {tie:.2f}%"


def test_radius_sentence_reflects_gqa_availability():
    """The radius sentence must track whether the harness supports GQA (gpu-sim-g8wc).

    Guard: fail if GQA support is lost (regression). The sentence should state that the CP term can now
    be checked because the harness supports grouped-query attention.
    """
    src = (ROOT / "tools" / "torch_parallel").rglob("*.py")
    has_gqa = any("n_kv_heads" in p.read_text() for p in src)
    assert has_gqa, "GQA support was lost from tools/torch_parallel (gpu-sim-g8wc regression)"
    para = _para("How wrong would the decisive term have to be")
    assert "supports grouped-query" in para, "the paragraph must say the harness supports GQA"
    assert "counterpart" in para, "the paragraph must say the decisive parameter has a counterpart"


def test_scaleout_paragraph_carries_the_computed_answers():
    """gpu-sim-scaleout: the beyond-a-rack claims must match what the script computes."""
    so = FACTS["scaleout"]
    para = _para("Beyond one rack")
    assert f"{so['n70b_80gpus_ring_margin']:.2f}%" in para, "missing the 80-GPU ring margin"
    assert f"{so['n70b_80gpus_ag_margin']:.2f}%" in para, "missing the 80-GPU default-convention margin"
    assert f"{so['worst_margin_8000']:.2f}%" in para and f"{so['best_margin_8000']:.2f}%" in para, (
        "the 8000-GPU margins must be quoted as a range")
    assert so["n405b_80gpus_feasible"] is False, "405B is feasible at 80 GPUs now: rewrite the paragraph"
    # The reason must be the one that is actually binding. An earlier version blamed the absence of
    # activation recomputation; at 80 GPUs the optimizer state alone is 70.9 GB, so recompute cannot help.
    assert "optimizer state" in para, "the paragraph must name the binding term (optimizer state)"
    assert f"{so['opt_state_floor_gb_405b_at_80']:.2f} GB" in para or f"{so['opt_state_floor_gb_405b_at_80']:.1f} GB" in para, (
        "the paragraph must carry the computed optimizer floor at 80 GPUs")
    assert "recomputation cannot close a gap" in para.lower(), (
        "the paragraph must say recomputation is not the fix at this scale")


def test_bubble_default_is_stated_as_pessimistic_and_winners_are_stable():
    """gpu-sim-bubble: micro_batches=pp is a pessimistic default; winners must be stable over 1x-8x."""
    mb = FACTS["micro_batches_sensitivity"]
    para = _para("The PP bubble default is pessimistic")
    stable = 0
    for n, rows in mb.items():
        shapes = {tuple(rows[k]["config"]) for k in ("x1", "x2", "x4", "x8")}
        if len(shapes) == 1:
            stable += 1
        for k in ("x1", "x8"):
            assert f"{rows[k]['step_ms']:.2f}" in para, (
                f"N={n}: paragraph is missing the {k} step time {rows[k]['step_ms']:.2f}")
    assert stable == 3, f"only {stable}/3 GPU counts keep their winner from 1x to 8x: rewrite the paragraph"
    # the direction claim: more micro-batches must never make the winner more expensive
    for n, rows in mb.items():
        assert rows["x4"]["step_ms"] <= rows["x1"]["step_ms"], (
            f"N={n}: raising micro_batches made the winner slower, so 'pessimistic' is the wrong word")


def test_headline_states_both_conventions_for_every_cell():
    """gpu-sim-9oz8: the answer depends on the CP rotate method everywhere cp>1 is used."""
    conv = FACTS["cp_convention_answers"]
    start = DOC.index("Only ONE recommended config survives both conventions")
    para = DOC[start:DOC.index("**Robustness radius", start)]
    flipped = 0
    for cell, rows in conv.items():
        ring, ag = rows["ring_all_to_all"], rows["pytorch_default_all_gather"]
        for row in (ring, ag):
            assert f"{row['step_ms']:.2f} ms" in para, f"{cell}: table is missing {row['step_ms']} ms"
            assert f"{row['margin_pct']:.2f}%" in para, f"{cell}: table is missing {row['margin_pct']}%"
        if ring["config"] != ag["config"]:
            flipped += 1
    assert flipped >= 3, f"only {flipped} cells differ between conventions: rewrite this table"
    # the convention-independent case must be the one with no context parallelism
    n16 = conv["N16_seq8192"]
    assert n16["ring_all_to_all"]["config"] == n16["pytorch_default_all_gather"]["config"]
    assert n16["ring_all_to_all"]["config"][2] == 1, "the convention-independent cell should be cp=1"
    assert "convention-INDEPENDENT" in para


def test_convention_gap_sentence_carries_the_measured_ratio_and_the_consequence():
    """gpu-sim-cpgqa: the CP convention gap is measured, and it exceeds the radius."""
    gaps = list(FACTS["cp_convention_gap"].values())
    para = _para("the convention gap is 1.99x")
    for g in gaps:
        assert f"{g:.2f}x" in para, f"the paragraph does not carry the measured gap {g:.2f}x"
    radius = FACTS["sensitivity_radius"]["N8_seq8192"]["flip_cp"]
    assert min(gaps) > radius, (
        f"the measured gap {min(gaps):.2f}x is now inside the radius {radius}: re-derive the doc claim")
    assert "conditional" in para, "the paragraph must state that the verdict is conditional on convention"


def test_radius_sentence_carries_the_flip_factors():
    """Both verdicts must publish how wrong their decisive term can be."""
    rad = FACTS["sensitivity_radius"]
    para = _para("How wrong would the decisive term have to be")
    for cell, key in (("N8_seq8192", "flip_cp"), ("N16_seq8192", "flip_pp")):
        k = rad[cell][key]
        assert k is not None, f"{cell} {key} is None: the term is no longer decisive, rewrite this test"
        assert f"{k:.2f}x" in para, f"the radius paragraph does not carry {cell} {key} = {k:.2f}x"


def test_cp_crossover_sentence_carries_the_bisected_crossover():
    cross = FACTS["cp_crossover"]
    para = _para("Where does CP actually start paying")
    assert str(cross["implied_seq"]) in para, "crossover sentence lacks the implied-MFU crossover"
    assert str(cross["measured_seq"]) in para, "crossover sentence lacks the measured-MFU crossover"
    assert f"{cross['advantage_at_8192_implied_pct']:.2f}%" in para


def test_headroom_sentence_names_the_config_the_model_now_picks():
    cell = FACTS["headroom_sensitivity"]["grid"]["N16_seq32768"]["hr0.10"]
    para = _para("Two cells recommend a config nobody would deploy")
    tp, pp, cp, dp, zero = cell["config"]
    expected = f"(tp={tp} pp={pp} cp={cp} dp={dp} z{zero})"
    assert expected in para, f"headroom paragraph should name {expected}"
    assert f"{cell['step_ms']:.2f}" in para, "headroom paragraph lacks the corrected step time"


def test_recommendation_block_carries_the_published_step_time():
    """The headline number is recomputed here, not trusted from a previous run."""
    model = dict(ps.MODEL)
    model["seq"] = 8192
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": "specs/h100_rack.json"}
    _, feas = ps.search(8, hw, model)
    best = feas[0]
    assert (best.tp, best.pp, best.cp, best.dp, best.zero) == (4, 1, 2, 1, 0)
    headline = f"{best.step_sum * 1e3:.2f} ms"
    assert headline in DOC, f"the published step time {headline} is not in the document"
