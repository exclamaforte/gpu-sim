"""gpu-sim-a5ro: total_params() is GQA-aware and matches published counts.

The old formula charged K/V at full width (4*d*d) with one embedding -- +15.6% at
405B, +16.2% at 70B. The fix (experiments/gpu-sim-a5ro/reference_params.json) charges
k,v at kv_dim and counts both untied embeddings.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import parallelism_search as ps

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / "experiments" / "gpu-sim-a5ro" / "reference_params.json").read_text())


def _m8():
    return dict(ps.MODEL)  # llama3-8b


def _m405():
    m = dict(ps.MODEL)
    m.update({"n_layers": 126, "d_model": 16384, "n_heads": 128, "n_kv_heads": 8,
              "head_dim": 128, "d_ff": 53248, "vocab": 128256})
    return m


def test_matches_published_counts():
    # bead acceptance: within 1% of the real published counts.
    assert abs(ps.total_params(_m8()) - 8.03e9) / 8.03e9 < 0.01, ps.total_params(_m8())
    assert abs(ps.total_params(_m405()) - 405.8e9) / 405.8e9 < 0.01, ps.total_params(_m405())


def test_matches_reference_table():
    # the committed reference table is the convention; the model must reproduce its
    # two-embedding column (llama2-34b has no official count and is excluded by design).
    assert abs(ps.total_params(_m8()) - REF["llama3-8b"]["two_embeddings"] * 1e9) / 8.03e9 < 0.01
    assert abs(ps.total_params(_m405()) - REF["llama3-405b"]["two_embeddings"] * 1e9) / 405.8e9 < 0.01


def test_gqa_aware_kv_width():
    # the defect was K/V at full width: varying n_kv_heads must move the count, and
    # full-width kv must cost exactly the old 4*d*d attention term.
    m = _m405()
    p_gqa = ps.total_params(m)
    m_mha = dict(m, n_kv_heads=128)
    p_mha = ps.total_params(m_mha)
    assert p_gqa < p_mha
    L, d = m["n_layers"], m["d_model"]
    assert p_mha - p_gqa == L * 2 * d * (d - 8 * 128), (p_mha, p_gqa)


def test_untied_embeddings_counted_twice():
    # Llama-3 does not tie: dropping one embedding must lose exactly V*d.
    m = _m8()
    assert ps.total_params(m) - (32 * (2 * 4096 * 4096 + 2 * 4096 * 1024 + 3 * 4096 * 14336)
                                 + 128256 * 4096) == 128256 * 4096
