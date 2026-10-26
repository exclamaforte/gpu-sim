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


def test_n16_sentence_says_it_is_a_tie_with_current_numbers():
    grid = FACTS["fragility_map"]["grid"]["N16_seq8192"]
    para = _para("N=16 is not a verdict at any efficiency")
    for tag in ("implied", "measured", "typical"):
        val = f"{grid[tag]['margin_pct']:.2f}%"
        assert val in para, f"N=16 paragraph does not carry {tag} margin {val}"
    assert "tie" in para.lower() or "does not distinguish" in para.lower()


def test_cp_crossover_sentence_carries_the_bisected_crossover():
    cross = FACTS["cp_crossover"]
    para = _para("Where does CP actually start paying")
    assert str(cross["implied_seq"]) in para, "crossover sentence lacks the implied-MFU crossover"
    assert str(cross["measured_seq"]) in para, "crossover sentence lacks the measured-MFU crossover"
    assert f"{cross['advantage_at_8192_implied_pct']:.2f}%" in para


def test_headroom_sentence_names_the_config_the_model_now_picks():
    cell = FACTS["headroom_sensitivity"]["grid"]["N16_seq32768"]["hr0.10"]
    para = _para("And one cell recommends a config nobody would deploy")
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
