"""b5f2: the radius paragraph must track the two-sided artifact (no 'never flips')."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "training-parallelism-sim.md").read_text()
FACTS = json.loads((ROOT / "docs" / "model_facts.json").read_text())


def _para(anchor):
    i = DOC.index(anchor)
    start = DOC.rfind("\n\n", 0, i) + 2
    end = DOC.find("\n\n", i)
    return DOC[start:end if end > 0 else len(DOC)]


def test_radius_paragraph_carries_n8_cp_both_sides_from_artifact():
    rad = FACTS["sensitivity_radius"]["N8_seq8192"]
    flip_down = rad["flip_cp_down"]
    flip_up = rad["flip_cp_up"]
    para = _para("Robustness radius")
    assert f"{flip_down:.4g}" in para, f"missing downward bound {flip_down:.4g}"
    assert f"{flip_up:.4g}" in para, f"missing upward bound {flip_up:.4g}"


def test_radius_paragraph_carries_n16_pp_up_from_artifact():
    rad = FACTS["sensitivity_radius"]["N16_seq8192"]
    pp_up = rad["flip_pp_up"]
    pp_down = rad["flip_pp_down"]
    assert pp_down is None, "N16 pp_down should be None"
    para = _para("Robustness radius")
    assert f"{pp_up:.3g}" in para, f"missing N16 pp upward bound {pp_up:.3g}"


def test_radius_paragraph_n16_clause_has_no_n8_pp_value():
    # slice on 'the N=16 verdict' so a correct mention of 0.8255 for N8 is allowed
    idx = DOC.index("the N=16 verdict")
    clause = DOC[idx:idx + 400]
    assert "0.8255" not in clause, "N16 clause must not carry N8's pp_down 0.8255"
    assert "0.83x" not in clause, "N16 clause must not carry N8's 0.83x"


def test_radius_paragraph_no_sentinel_lower_bound():
    assert "[0, 1.52]" not in DOC
    assert "[0,1.52]" not in DOC


def test_radius_paragraph_no_never_flips():
    assert "never flips" not in DOC.lower()


def test_radius_paragraph_built_from_artifact_not_typed():
    rad = FACTS["sensitivity_radius"]["N8_seq8192"]
    flip_down = rad["flip_cp_down"]
    flip_up = rad["flip_cp_up"]
    assert f"{flip_down:.4g}" in DOC
    assert f"{flip_up:.4g}" in DOC
    rad16 = FACTS["sensitivity_radius"]["N16_seq8192"]
    pp_up = rad16["flip_pp_up"]
    assert f"{pp_up:.3g}" in DOC


def test_n16_pp_down_is_unbounded_not_sweep_floor():
    rad = FACTS["sensitivity_radius"]["N16_seq8192"]
    assert rad["flip_pp_down"] is None
    assert "unbounded below" in DOC, "N16 pp has no downward flip: must say 'unbounded below'"
    assert "[0.05x, 1.86x]" not in DOC, "sweep floor must not be published as a measured bound"


import re as _re

def _slice(anchor, span=400):
    i = DOC.index(anchor)
    return DOC[i:i + span]


def test_n8_cp_para_no_borrowed_pp_bound():
    """Paragraph-scoped: the CP lower bound at :42 must be 0.5415, not the borrowed 0.8255."""
    para = _para("Robustness radius")
    assert "0.5415" in para
    m = _re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", para)
    assert m is not None
    assert m.group(1) == "0.5415", f"CP lower bound is {m.group(1)}, expected 0.5415"


def test_n16_clause_bans_borrowed_and_083x():
    """Clause-scoped ban (both 0.8255 and 0.83x) inside the N=16 verdict clause."""
    clause = _slice("the N=16 verdict")
    assert "0.8255" not in clause
    assert "0.83x" not in clause


def test_n16_pp_unbounded_below_in_clause():
    """The unbounded-below claim must live in the N=16 clause, not elsewhere."""
    clause = _slice("the N=16 answer for PP-bubble")
    assert "unbounded below" in clause


def test_n8_cp_seq32768_interval_present():
    rad = FACTS["sensitivity_radius"]["N8_seq32768"]
    lo = f"{rad['flip_cp_down']:.2f}x"
    hi = f"{rad['flip_cp_up']:.2f}x"
    assert f"[{lo}, {hi}]" in DOC


def test_discovery_of_two_number_intervals():
    """Widened discovery: every 2-number bracket must be registered."""
    found = set(_re.findall(r"\[\s*[\d.]+x?\s*,\s*[\d.]+x?\s*\]", DOC))
    allowed = {
        "[0.5415, 1.5244]",
        "[0.54x, 1.52x]",
        "[0.20x, 4.81x]",
        "[1, 8]",
        "[1,8]",
        "[2, 3]",
        "[2,3]",
    }
    unexpected = found - allowed
    assert not unexpected, f"unregistered 2-number interval(s): {unexpected}"


def test_n8_cp_para_flip_configs_guarded():
    para = _para("Robustness radius")
    assert "(2,1,4,1,0)" in para
    assert "(1,8,1,1,0)" in para
