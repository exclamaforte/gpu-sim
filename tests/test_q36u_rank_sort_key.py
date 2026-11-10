"""gpu-sim-q36u: assert the ranking claim on synthetic rows, cheaply.

The 20ms row bound this claim to test_error_budget_ranking.test_top_row, which
runs the whole ranking (~177s) and the walk runs it TWICE (mutated + restored):
~351s against the 300s per-test budget, so the suite went red by arithmetic.
The claim itself is a sort-key statement, so it is asserted here directly.
"""

from scripts.error_budget import rank_sort_key


def test_one_sided_margin_drives_the_order():
    # A: worse one-sided margin (0.10) but strong flip_down; B: better margin.
    a = {"fraction_one_sided": 0.10, "flip_down": 0.90, "elasticity": 1.0}
    b = {"fraction_one_sided": 0.50, "flip_down": 0.10, "elasticity": 1.0}
    assert sorted([a, b], key=rank_sort_key)[0] is a


def test_rows_without_a_margin_sort_last_by_current_margin():
    no_margin = {"fraction_one_sided": None, "current_margin": 0.2}
    with_margin = {"fraction_one_sided": 0.9, "elasticity": 1.0}
    assert sorted([no_margin, with_margin], key=rank_sort_key) == [with_margin, no_margin]


def test_infinite_elasticity_wins_the_tie_break():
    # Infinite elasticity is the THIRD tuple element: it only decides rows with
    # the same one-sided fraction. My first version of this test asserted it
    # sorted first overall, which the implementation never promised.
    inf = {"fraction_one_sided": 0.5, "elasticity": None, "elasticity_infinite": True}
    finite = {"fraction_one_sided": 0.5, "elasticity": 2.0}
    assert sorted([finite, inf], key=rank_sort_key)[0] is inf
