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
