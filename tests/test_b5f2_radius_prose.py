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


def test_radius_paragraph_carries_both_sides_from_artifact():
    rad = FACTS["sensitivity_radius"]["N8_seq8192"]
    flip_down = rad["flip_cp_down"]
    flip_up = rad["flip_cp_up"]
    para = _para("Robustness radius")
    lo = f"{flip_down:.4g}"
    hi = f"{flip_up:.4g}"
    assert lo in para, f"radius paragraph missing downward bound {lo}"
    assert hi in para, f"radius paragraph missing upward bound {hi}"
    assert "two-sided" in para.lower() or "underestimate" in para.lower(), (
        "radius paragraph must name the downward side")


def test_radius_paragraph_no_sentinel_lower_bound():
    assert "[0, 1.52]" not in DOC
    assert "[0,1.52]" not in DOC


def test_radius_paragraph_no_never_flips():
    assert "never flips" not in DOC.lower()


def test_radius_paragraph_built_from_artifact_not_typed():
    rad = FACTS["sensitivity_radius"]["N8_seq8192"]
    flip_down = rad["flip_cp_down"]
    flip_up = rad["flip_cp_up"]
    lo = f"{flip_down:.4g}"
    hi = f"{flip_up:.4g}"
    assert lo in DOC, f"artifact-derived lower bound {lo} not found in doc"
    assert hi in DOC, f"artifact-derived upper bound {hi} not found in doc"
