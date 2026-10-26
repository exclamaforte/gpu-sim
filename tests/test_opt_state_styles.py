"""Optimizer-state styles (gpu-sim-scup, v2: steady/peak split per PI 18:44 comment).

Rig table (experiments/gpu-sim-recompute/measure_opt_styles.py + fused-peak runs),
B/param TOTALS including resident params; the model reproduces each cell through
its sharded components (params full, grads ZeRO-2, master/moments/temps ZeRO-1):

  style          params  grads  master  moments  temps(dflt/fused)  total dflt/fused/steady
  fp32_master    2       2      4       8        0/0                14 / 14 / 14 (flat)
  bf16_moments   2       2      0       4        2/2                10 / 10 / 6
  fp32_moments   4       2      0       8        6/2                20 / 16 / 12

fp32_master is the legacy flat 14 at EVERY opt_peak (approximates fp32_moments
steady + 2 margin, underestimates peak) so the default path is bit-identical.
'none' prices steady only (grads + temps freed) and is OPTIMISTIC.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve(__file__).parents[1]))
from epic_cluster import parallelism_search as ps


def _hw(**kw):
    hw = {"peak_flops": 989e12, "mem_gb": 80.0, "spec_path": "specs/h100_rack.json"}
    hw.update(kw)
    return hw


def _p_shard(m, tp=4, pp=1):
    return ps.total_params(m) / (tp * pp) / 1e9


def test_default_is_fp32_master_and_bit_identical_at_every_peak_mode():
    """No key, explicit fp32_master, and every opt_peak agree: default unmoved."""
    model = dict(ps.MODEL)
    for n in (8, 16):
        _, base = ps.search(n, _hw(), model)
        for st, pk in (("fp32_master", "default"), ("fp32_master", "fused"),
                       ("fp32_master", "none")):
            _, got = ps.search(n, _hw(opt_state=st, opt_peak=pk), model)
            assert [(c.tp, c.pp, c.cp, c.dp, c.zero) for c in base] == \
                   [(c.tp, c.pp, c.cp, c.dp, c.zero) for c in got]
            for b, g in zip(base, got):
                assert b.step_sum == g.step_sum and b.memory_gb == g.memory_gb


def test_style_peak_totals_match_the_rig_table():
    """End-to-end memory deltas at dp=1 equal the rig peak totals minus default 16."""
    model = dict(ps.MODEL)
    base = ps.score(4, 1, 1, 1, 0, _hw(), model).memory_gb
    cases = [("bf16_moments", "default", 10 - 16),
             ("fp32_moments", "default", 20 - 16),
             ("fp32_moments", "fused", 16 - 16),
             ("bf16_moments", "none", 6 - 16),
             ("fp32_moments", "none", 12 - 16)]
    for st, pk, delta_b in cases:
        got = ps.score(4, 1, 1, 1, 0, _hw(opt_state=st, opt_peak=pk), model).memory_gb
        assert got - base == pytest.approx(delta_b * _p_shard(model), rel=0.01), \
            f"{st}/{pk}"


def test_fused_and_default_differ_in_composition_not_just_total():
    """fp32_moments default and fused both total 16 B/param: the breakdown tells them apart."""
    model = dict(ps.MODEL)
    d = ps.score(4, 1, 1, 1, 0, _hw(opt_state="fp32_moments", opt_peak="default"), model)
    f = ps.score(4, 1, 1, 1, 0, _hw(opt_state="fp32_moments", opt_peak="fused"), model)
    assert d.memory_gb - f.memory_gb == pytest.approx(4 * _p_shard(model), rel=0.01)
    assert d.mem_temps_gb == pytest.approx(6 * _p_shard(model), rel=0.01)
    assert f.mem_temps_gb == pytest.approx(2 * _p_shard(model), rel=0.01)
    assert d.mem_master_gb == 0.0 and f.mem_master_gb == 0.0


def test_unknown_style_or_peak_raises_instead_of_silently_pricing_fp32():
    model = dict(ps.MODEL)
    with pytest.raises(ValueError):
        ps.score(4, 1, 2, 1, 0, _hw(opt_state="fp8_someday"), model)
    with pytest.raises(ValueError):
        ps.score(4, 1, 2, 1, 0, _hw(opt_state="fp32_moments", opt_peak="foreach-off"), model)
