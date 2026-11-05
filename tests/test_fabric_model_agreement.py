"""Chain test: the ENFORCED fabric delay must equal the MODELLED collective time.

Why this file exists (gpu-sim-3v6d.1)
-------------------------------------
On 2026-11-04 one NVLink link carried three different bandwidths:

    900   specs/h100_rack.json      bw_gbps, declared bw_is_bidirectional
    450   topology.emit + nccl_cost eff_bw_gbps for the same link
    112.5 fakegpu_fabric.c:730     nbytes/(900*125) us -- a Gbit/s reading

The model priced 450 GB/s, the enforcer acted at 112.5 GB/s, and every test
passed, because the two numbers were never put in the same assertion: the
agreement test (test_model_agreement.py) compares C-predict to Python-predict
and never drives the transport, while the fabric tests check delivery, FIFO
order and identity, never magnitude.

This is the same failure shape tests/test_chain_spec_to_sleep.py was written
for on the compute side (spec -> table -> env -> parse -> sleep): each link
was tested in isolation and the chain was not. This test walks

    spec JSON -> topology.emit -> eff_bw_gbps -> fg_fab_transfer -> slept time

in one process, and asserts the slept time against the same formula
fakegpu_nccl.c prices collectives with. A plant that passes the raw
bidirectional rate, or that reinterprets GB/s as Gbit/s, fails it by 2x/8x.
"""
import ctypes
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from epic_cluster import topology  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(REPO, "libfakegpu.so")
SPEC = os.path.join(REPO, "specs", "h100_rack.json")

PAYLOAD = 4 << 30  # 4 GiB: at 450 GB/s that is 9.5 ms, so usleep overshoot (~60 us)
                   # is <1% and the 2x/8x plants stay far outside any sane tolerance
TOL = 0.02  # acceptance (a): 2%. Measured +1.15% at the tip (reviewer, 2026-11-04);
           # 0.05 was a loosening that made the test stop enforcing the written bar.


def _load_topo(tmp_path):
    import json
    with open(SPEC) as f:
        spec = json.load(f)
    blob = topology.emit(spec)
    path = tmp_path / "h100_rack.bin"
    path.write_bytes(blob)
    return spec, str(path)


def _attach(lib, run_id, nranks=2):
    lib.fg_fab_attach.restype = ctypes.c_void_p
    lib.fg_fab_attach.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_uint64,
                                  ctypes.POINTER(ctypes.c_int)]
    err = ctypes.c_int(0)
    f = lib.fg_fab_attach(run_id.encode(), 0, 0, nranks, 0, ctypes.byref(err))
    assert f, "fg_fab_attach failed err=%d" % err.value
    lib.fg_fab_transfer.restype = ctypes.c_int
    lib.fg_fab_transfer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
                                    ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                    ctypes.c_int, ctypes.c_uint64, ctypes.c_double,
                                    ctypes.c_int]
    return f


def _measure_us(lib, f, nbytes, bw_gbs, lat_us):
    """Wall time of one modeled transfer, best of 3 (first call is cold)."""
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        rc = lib.fg_fab_transfer(f, 0, nbytes, bw_gbs, lat_us, 0.0, 0, 0, 0.0, -1)
        dt = (time.perf_counter() - t0) * 1e6
        assert rc == 0, rc
        best = dt if best is None else min(best, dt)
    return best


def _nvlink_terms(spec):
    link = next(l for l in spec["links"] if l["kind"] == "nvlink")
    eff = topology.effective_bw(link, topology._shared_counts(spec)
                                if hasattr(topology, "_shared_counts") else {})
    return link, eff


@pytest.fixture(scope="module")
def fab():
    lib = ctypes.CDLL(LIB)
    f = _attach(lib, "chain-%d" % os.getpid())
    yield lib, f
    lib.fg_fab_detach.argtypes = [ctypes.c_void_p]
    lib.fg_fab_detach(f)


def test_effective_rate_is_half_the_bidirectional_spec_value(tmp_path):
    """The spec declares 900 aggregate bidirectional; one-way usable is 450."""
    spec, _ = _load_topo(tmp_path)
    link = next(l for l in spec["links"] if l["kind"] == "nvlink")
    assert link["bw_gbps"] == 900.0
    assert link.get("bw_is_bidirectional") is True
    eff = topology.effective_bw(link, {})
    assert eff == 450.0, eff


def test_enforced_delay_matches_model_formula(tmp_path, fab):
    """spec -> emit -> eff_bw -> fg_fab_transfer -> slept time == model term."""
    lib, f = fab
    spec, _ = _load_topo(tmp_path)
    link, eff = _nvlink_terms(spec)
    lat_us = float(link["latency_us"])

    measured = _measure_us(lib, f, PAYLOAD, eff, lat_us)
    model = PAYLOAD / (eff * 1e9) * 1e6 + lat_us       # fakegpu_nccl.c's formula
    rel = abs(measured - model) / model
    assert rel <= TOL, (
        "enforced %.1f us vs modelled %.1f us (%.1f%% off) "
        "bw_gbs=%.1f -- the transport is not charging what the model prices"
        % (measured, model, 100 * rel, eff))


def test_plant_raw_bidirectional_rate_fails_the_same_assertion(tmp_path, fab):
    """NEGATIVE CONTROL: passing the raw 900 (bidir) instead of 450 is 2x fast.

    If this plant ever PASSES the assertion above, the test has stopped
    discriminating between the spec value and the effective one.
    """
    lib, f = fab
    spec, _ = _load_topo(tmp_path)
    link, eff = _nvlink_terms(spec)
    lat_us = float(link["latency_us"])

    raw = float(link["bw_gbps"])
    measured = _measure_us(lib, f, PAYLOAD, raw, lat_us)
    model = PAYLOAD / (eff * 1e9) * 1e6 + lat_us
    rel = abs(measured - model) / model
    assert rel > TOL, "plant with raw %.1f was NOT detected (%.1f%%)" % (raw, 100 * rel)
    # and it is exactly the bidirectional halving, not some other error
    assert abs(measured / model - 0.5) < 0.05, measured / model


def test_plant_gbit_convention_is_eight_times_fast(tmp_path, fab):
    """NEGATIVE CONTROL: reading GB/s as Gbit/s divides the rate by 8.

    This is the convention fakegpu_fabric.c:730 shipped with until
    gpu-sim-3v6d.1: the same field, read as Gbit/s.
    """
    lib, f = fab
    spec, _ = _load_topo(tmp_path)
    link, eff = _nvlink_terms(spec)
    lat_us = float(link["latency_us"])

    as_gbits = eff / 8.0
    measured = _measure_us(lib, f, PAYLOAD, as_gbits, lat_us)
    model = PAYLOAD / (eff * 1e9) * 1e6 + lat_us
    assert abs(measured / model - 8.0) < 0.2, measured / model


def test_one_rate_everywhere_on_the_real_spec(tmp_path):
    """One link, one rate: the Python rule, the emitted binary, the C predictor.

    The C predictor prices with whatever `eff_bw_gbps` the binary carries
    (fakegpu_nccl.c:346), so this asserts the single number all three consumers
    see.  Before gpu-sim-3v6d.1 the emitted binary carried 900 (topology's copy
    of the rule never halved a bidirectional link) while link_model priced 450:
    a 2x split between the two predictors, invisible because
    tests/test_model_agreement.py uses a spec with no bidirectional flag.
    """
    from epic_cluster import link_model
    import json

    spec, _ = _load_topo(tmp_path)
    counts = topology.shared_group_counts(spec)
    link = next(l for l in spec["links"] if l["kind"] == "nvlink")

    py_rule = topology.effective_bw(link, counts)
    model_rule = link_model._eff_bw(link, counts)
    assert py_rule == 450.0, py_rule
    assert model_rule == py_rule, (model_rule, py_rule)   # delegated, not copied

    back = topology.read_binary(topology.emit(spec))
    nv = [l["eff_bw_gbps"] for l in back["links"] if l["bw_gbps"] == 900.0]
    assert nv and all(e == 450.0 for e in nv), sorted(set(nv))
    assert max(l["eff_bw_gbps"] for l in back["links"]) == 450.0

    # PLANT: drop the declaration and every consumer moves together to 900.
    # If this stops being 2x, the halving stopped being driven by the flag.
    other = dict(link)
    other.pop("bw_is_bidirectional", None)
    assert topology.effective_bw(other, counts) == 900.0
    assert link_model._eff_bw(other, counts) == 900.0

import unittest  # imported here for module-level placement (gpu-sim-05b3)

class TestFabricModelAgreement(unittest.TestCase):
    """Class-wrapped form of the chain assertion above (gpu-sim-3v6d.1).

    Exists so the liveness ledger can name the enforcing assertion as
    module.Class.method: the ledger normalises a 2-part id to
    tests/<module>.py::<method> and a 3-part id to ::<Class>::<method>, and a
    class keeps the row valid under both readers. No fixtures, so a replay that
    imports and calls it works too. Same measurement as
    test_enforced_delay_matches_model_formula -- if this stops agreeing with the
    model formula, the transport is not charging what nccl_cost prices.
    """

    def test_enforced_delay_matches_model_formula_class(self):
        import pathlib
        import shutil
        import tempfile

        d = pathlib.Path(tempfile.mkdtemp(prefix="fab-agree-"))
        try:
            spec, _ = _load_topo(d)
            link, eff = _nvlink_terms(spec)
            assert eff == 450.0, eff
            lat_us = float(link["latency_us"])
            lib = ctypes.CDLL(LIB)
            f = _attach(lib, "chain-class-%d" % os.getpid())
            try:
                measured = _measure_us(lib, f, PAYLOAD, eff, lat_us)
            finally:
                lib.fg_fab_detach.argtypes = [ctypes.c_void_p]
                lib.fg_fab_detach(f)
            model = PAYLOAD / (eff * 1e9) * 1e6 + lat_us   # fakegpu_nccl.c's formula
            rel = abs(measured - model) / model
            assert rel <= TOL, (
                "enforced %.1f us vs modelled %.1f us (%.1f%% off) "
                "bw_gbs=%.1f -- the transport is not charging what the model prices"
                % (measured, model, 100 * rel, eff))
        finally:
            shutil.rmtree(d, ignore_errors=True)