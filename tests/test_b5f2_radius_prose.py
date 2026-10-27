"""b5f2: the radius paragraph must track the two-sided artifact (no 'never flips')."""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "training-parallelism-sim.md").read_text()
FACTS = json.loads((ROOT / "docs" / "model_facts.json").read_text())


def _slice(anchor, span=500):
    i = DOC.index(anchor)
    return DOC[i:i + span]


def _interval_strings():
    """The exact interval-shaped bracket strings the doc is allowed to carry."""
    rad8 = FACTS["sensitivity_radius"]["N8_seq8192"]
    rad8b = FACTS["sensitivity_radius"]["N8_seq32768"]
    rad16 = FACTS["sensitivity_radius"]["N16_seq8192"]
    lo8 = f"{rad8['flip_cp_down']:.2f}x"
    hi8 = f"{rad8['flip_cp_up']:.2f}x"
    lo8b = f"{rad8b['flip_cp_down']:.2f}x"
    hi8b = f"{rad8b['flip_cp_up']:.2f}x"
    hi16 = f"{rad16['flip_pp_up']:.2f}x"
    return {
        f"[{lo8}, {hi8}]",
        f"[{lo8b}, {hi8b}]",
        "[unbounded below (no flip down to the 0.05x sweep floor), 1.86x upward]",
        "[unbounded below, 1.86x]",
    }


def test_n8_cp_seq8192_interval_present():
    rad = FACTS["sensitivity_radius"]["N8_seq8192"]
    lo = f"{rad['flip_cp_down']:.2f}x"
    hi = f"{rad['flip_cp_up']:.2f}x"
    assert f"[{lo}, {hi}]" in DOC


def test_n8_cp_seq32768_interval_present():
    rad = FACTS["sensitivity_radius"]["N8_seq32768"]
    lo = f"{rad['flip_cp_down']:.2f}x"
    hi = f"{rad['flip_cp_up']:.2f}x"
    assert f"[{lo}, {hi}]" in DOC


def test_n16_pp_interval_unbounded_below():
    rad = FACTS["sensitivity_radius"]["N16_seq8192"]
    assert rad["flip_pp_down"] is None
    assert "unbounded below" in DOC
    assert "[0.05x, 1.86x]" not in DOC


def test_n16_pp_up_fragment_present():
    rad = FACTS["sensitivity_radius"]["N16_seq8192"]
    hi = f"{rad['flip_pp_up']:.2f}x"
    assert hi in _slice("the N=16 verdict") or hi in DOC


def test_no_sentinel_lower_bound():
    assert "[0, 1.52]" not in DOC
    assert "[0,1.52]" not in DOC


def test_no_never_flips():
    assert "never flips" not in DOC.lower()


def test_no_sweep_floor_as_bound():
    # the sweep floor must not be printed as a measured lower bound for N16 pp
    assert "[0.05x, 1.86x]" not in DOC


def test_discovery_of_x_intervals():
    # discover every bracket carrying an 'x' and require it be in the allow-list
    found = set(re.findall(r"\[[^\]]*x[^\]]*\]", DOC))
    allowed = _interval_strings()
    unexpected = found - allowed
    assert not unexpected, f"unregistered interval-shaped claim(s): {unexpected}"


def test_clause_scoped_unbounded():
    clause = _slice("the N=16 verdict")
    for bad in ("0.8255", "0.83x"):
        assert bad not in clause, f"N16 clause must not carry {bad}"
