"""Seq-dependent recompute alpha (gpu-sim-4qcx).

The model charged a constant alpha=0.67 (MHA block, seq 8192) at every sequence.
The GQA re-measure (experiments/gpu-sim-recompute/measure_ckpt_gqa.py) retains
0.824/0.580/0.406 at seq 2048/4096/8192, so the constant UNDER-predicted
activation memory below 8192 by up to 0.49 of the footprint (optimistic).

alpha(seq) is piecewise-linear through the three GQA points, flat 0.176 below
2048, and 0.67 at/above 8192 -- every published grid runs seq 8192, so the
default path is bit-identical. The joint at 8192 is a measurement-family
boundary, not physics.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from epic_cluster import parallelism_search as ps


def test_alpha_hits_the_three_rig_points_and_the_default_boundary():
    assert ps._recompute_alpha(2048) == pytest.approx(0.176)
    assert ps._recompute_alpha(4096) == pytest.approx(0.420)
    # 8192 takes the calibrated default, not the GQA 0.594: published grids run
    # seq 8192 and must not move.
    assert ps._recompute_alpha(8192) == pytest.approx(0.67)


def test_alpha_clamps_outside_the_measured_range():
    assert ps._recompute_alpha(512) == pytest.approx(0.176)
    assert ps._recompute_alpha(2048) == pytest.approx(0.176)
    assert ps._recompute_alpha(32768) == pytest.approx(0.67)
    assert ps._recompute_alpha(10**6) == pytest.approx(0.67)


def test_alpha_is_monotone_and_linear_between_points():
    assert ps._recompute_alpha(2048) < ps._recompute_alpha(3000) < ps._recompute_alpha(4096)
    assert ps._recompute_alpha(4096) < ps._recompute_alpha(6144) < ps._recompute_alpha(8192 - 1)
    assert ps._recompute_alpha(6144) == pytest.approx(0.420 + (0.594 - 0.420) * 0.5)


def _hw():
    return {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": "specs/h100_rack.json"}


def _discount(seq):
    """Activation memory shed by full recompute at a fixed config and seq."""
    m0 = dict(ps.MODEL, seq=seq)
    base = ps.score(4, 1, 1, 2, 0, _hw(), m0).memory_gb
    rec = ps.score(4, 1, 1, 2, 0, dict(_hw(), recompute=1.0), m0).memory_gb
    return base - rec


def test_short_seq_recompute_sheds_less_end_to_end():
    """The optimism fix through score(): the shed footprint is act(seq)*alpha(seq).

    act scales with seq (cp=1), so discount(2048)/discount(8192) must equal
    (2048/8192) * (0.176/0.67). A constant-alpha model gives 0.25 instead.
    """
    ratio = _discount(2048) / _discount(8192)
    assert ratio == pytest.approx(0.25 * (0.176 / 0.67), rel=0.02)
    assert ratio < 0.15, "short-seq recompute must shed much less than the constant claimed"
