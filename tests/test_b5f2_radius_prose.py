"""b5f2: the radius paragraph must track the two-sided artifact (no 'never flips')."""
import json
import pathlib
import re
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


def _slice(anchor, span=400):
    i = DOC.index(anchor)
    return DOC[i:i + span]


def _allowed_intervals():
    rad8 = FACTS["sensitivity_radius"]["N8_seq8192"]
    rad8b = FACTS["sensitivity_radius"]["N8_seq32768"]
    rad16 = FACTS["sensitivity_radius"]["N16_seq8192"]
    return {
        # flip claims (4-decimal paragraph form)
        f"[{rad8['flip_cp_down']:.4f}, {rad8['flip_cp_up']:.4f}]",
        # flip claims (2-decimal x form, doc:237)
        f"[{rad8['flip_cp_down']:.2f}x, {rad8['flip_cp_up']:.2f}x]",
        f"[{rad8b['flip_cp_down']:.2f}x, {rad8b['flip_cp_up']:.2f}x]",
        # unbounded (doc:238 and claims register)
        "[unbounded below (no flip down to the 0.05x sweep floor), 1.86x upward]",
        "[unbounded below, 1.86x]",
        # allow-listed non-flip intervals
        "[1, 8]",
        "[1,8]",
        "[2, 3]",
        "[2,3]",
    }


# --- paragraph-scoped: the N=8 radius sentence (doc:41-44) ---

def test_n8_cp_para_carries_both_sides_4dp():
    rad = FACTS["sensitivity_radius"]["N8_seq8192"]
    para = _para("Robustness radius")
    assert f"{rad['flip_cp_down']:.4g}" in para
    assert f"{rad['flip_cp_up']:.4g}" in para


def test_n8_cp_para_flip_configs_guarded():
    # the flip configs at :43 must be present and tied to the artifact
    para = _para("Robustness radius")
    assert "(2,1,4,1,0)" in para
    assert "(1,8,1,1,0)" in para


def test_n8_cp_para_no_borrowed_pp_bound():
    # 0.8255 is N8's pp_down; it must not appear as the CP lower bound at :42
    para = _para("Robustness radius")
    # the CP lower bound line must contain 0.5415, not 0.8255
    assert "0.5415" in para
    # ensure the first interval's lower bound is not the borrowed value
    import re
    m = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", para)
    assert m is not None
    assert m.group(1) == "0.5415", f"CP lower bound is {m.group(1)}, expected 0.5415"


# --- N8 seq32768 interval ---

def test_n8_cp_seq32768_interval_present():
    rad = FACTS["sensitivity_radius"]["N8_seq32768"]
    lo = f"{rad['flip_cp_down']:.2f}x"
    hi = f"{rad['flip_cp_up']:.2f}x"
    assert f"[{lo}, {hi}]" in DOC


# --- N16 pp interval ---

def test_n16_pp_unbounded_below_in_clause():
    rad = FACTS["sensitivity_radius"]["N16_seq8192"]
    assert rad["flip_pp_down"] is None
    clause = _slice("the N=16 answer for PP-bubble")
    assert "unbounded below" in clause, "N16 pp has no downward flip: must say 'unbounded below' in its clause"


def test_n16_pp_clause_no_borrowed_values():
    clause = _slice("the N=16 answer for PP-bubble")
    for bad in ("0.8255", "0.83x", "0.54x"):
        assert bad not in clause, f"N16 clause must not carry {bad}"


def test_n16_pp_up_fragment_in_clause():
    rad = FACTS["sensitivity_radius"]["N16_seq8192"]
    hi = f"{rad['flip_pp_up']:.2f}x"
    clause = _slice("the N=16 verdict")
    # the up fragment must be in the clause (not just anywhere in DOC)
    assert hi in clause or "1.86x" in clause


def test_n16_pp_no_sweep_floor_as_bound():
    assert "[0.05x, 1.86x]" not in DOC


# --- global bans ---

def test_no_sentinel_lower_bound():
    assert "[0, 1.52]" not in DOC
    assert "[0,1.52]" not in DOC


def test_no_never_flips():
    assert "never flips" not in DOC.lower()


# --- discovery: 2-number interval shapes ---

def test_discovery_of_two_number_intervals():
    found = set(re.findall(r"\[\s*[\d.]+x?\s*,\s*[\d.]+x?\s*\]", DOC))
    allowed = _allowed_intervals()
    # filter allowed to those matching the 2-number shape
    allowed_2num = {a for a in allowed if re.fullmatch(r"\[\s*[\d.]+x?\s*,\s*[\d.]+x?\s*\]", a)}
    unexpected = found - allowed_2num
    assert not unexpected, f"unregistered 2-number interval(s): {unexpected}"


def test_clause_scoped_unbounded():
    clause = _slice("the N=16 verdict")
    # unbounded claim must live in the N16 clause
    # (covered by test_n16_pp_unbounded_below_in_clause; here ensure no escape)
    assert "0.8255" not in clause
