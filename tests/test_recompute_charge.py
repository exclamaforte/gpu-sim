"""gpu-sim-hz3u: the recompute knob charges measured cost and keeps measured memory.

The knob is hw['recompute'], a float 0.0..1.0 defaulting to 0.0 (legacy "none"/"full"
strings map to 0.0/1.0; anything else raises). At 1.0 the model keeps (1-0.67) of the
stored activations and scales the compute term by 1.25 -- both numbers MEASURED on the
lab 5090 (experiments/gpu-sim-recompute/), not rules of thumb.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps


def _hw(rec):
    return {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
            "batch": 1, "recompute": rec}


def _cfg():
    m = dict(ps.MODEL)
    m["seq"] = 2048
    return (2, 1, 1, 1, 0), m


def test_default_is_zero_and_bit_identical():
    # omitting the knob is exactly recompute=0.0: every published number is untouched.
    args, m = _cfg()
    c_absent = ps.score(*args, {"peak_flops": 989e12, "mem_gb": 80.0, "fabric": "nvlink4_h100"}, m)
    c_zero = ps.score(*args, _hw(0.0), m)
    assert c_absent.compute == c_zero.compute
    assert c_absent.memory_gb == c_zero.memory_gb
    c_none = ps.score(*args, _hw("none"), m)
    assert c_none.compute == c_zero.compute
    assert c_none.memory_gb == c_zero.memory_gb


def test_full_charges_measured_1_25x_compute():
    args, m = _cfg()
    c0 = ps.score(*args, _hw(0.0), m)
    c1 = ps.score(*args, _hw(1.0), m)
    assert c1.compute == c0.compute * 1.25


def test_full_keeps_measured_0_33x_activations():
    # memory delta between r=0 and r=1 is exactly the 0.67 alpha on the activation term:
    # total(r=1) - total(r=0) == -(0.67 * act_bytes(r=0)), verified by recomputing act
    # from the model's own geometry (not by retyping a number).
    (tp, pp, cp, dp, zero), m = _cfg()
    c0 = ps.score(tp, pp, cp, dp, zero, _hw(0.0), m)
    c1 = ps.score(tp, pp, cp, dp, zero, _hw(1.0), m)
    assert c1.memory_gb < c0.memory_gb
    _dp_dtype = "bf16"
    dtype_bytes = 2
    d = m["d_model"]
    per_token_width = 3 * (d // tp) + m["d_ff"] // tp
    act0 = (m["n_layers"] / pp) * (m["seq"] / cp) * dtype_bytes * per_token_width * 2.3 / 1e9
    assert abs((c1.memory_gb - c0.memory_gb) - (-0.67 * act0)) < 1e-9


def test_legacy_strings_and_garbage():
    args, m = _cfg()
    c_full = ps.score(*args, _hw("full"), m)
    c_10 = ps.score(*args, _hw(1.0), m)
    assert c_full.compute == c_10.compute
    assert c_full.memory_gb == c_10.memory_gb
    with pytest.raises(ValueError):
        ps.score(*args, _hw("selective"), m)
    with pytest.raises(ValueError):
        ps.score(*args, _hw(1.5), m)
