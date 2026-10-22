"""Tests for cy0-L1: H100 rack spec + one-layer work model + compute + TP allreduce."""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cy0_l1
from epic_cluster.spec import load as load_spec, validate, SpecError


SPEC_PATH = REPO / "specs" / "h100_rack.json"


def test_spec_validates():
    spec = load_spec(str(SPEC_PATH))
    assert len(spec["nodes"]) == 2
    assert len(spec["ranks"]) == 16
    assert spec["nodes"][0]["devices"][0]["model"] == "h100_sxm"


def test_invalid_spec_rejected():
    bad = json.loads(SPEC_PATH.read_text())
    bad["models"]["h100_sxm"]["sm_count"] = 0
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bad, f)
        f.flush()
        try:
            validate(json.load(open(f.name)))
            assert False, "should have raised"
        except SpecError:
            pass
        finally:
            os.unlink(f.name)


def test_components_sum_to_total():
    result = cy0_l1.predict_layer(str(SPEC_PATH))
    total = sum(c["seconds"] for c in result["components"])
    assert abs(total - result["total_seconds"]) < 1e-12


def test_all_params_have_provenance():
    result = cy0_l1.predict_layer(str(SPEC_PATH))
    for c in result["components"]:
        assert c.get("provenance"), f"{c['op']} missing provenance"
        if c["kind"] == "compute":
            assert c.get("efficiency_provenance"), f"{c['op']} missing efficiency provenance"


def _predict_with_bandwidth(kind, factor, tp=4):
    spec = json.loads(SPEC_PATH.read_text())
    for link in spec["links"]:
        if link["kind"] == kind:
            link["bw_gbps"] *= factor
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        f.flush()
        try:
            return cy0_l1.predict_layer(f.name, tp=tp)
        finally:
            os.unlink(f.name)


def test_bandwidth_doubling_moves_only_the_fabric_in_use():
    """Structural test for the fabric the TP group actually crosses.

    After gpu-sim-by9 ranks are contiguous per node, so a TP group of 4 sits on
    ONE node and crosses NVLink. Two assertions, and the second is the one that
    catches a placement regression:
      - doubling NVLink bandwidth strictly reduces TP comm, compute unchanged;
      - doubling IB bandwidth does NOT change it, because TP never touches IB.
    """
    base = cy0_l1.predict_layer(str(SPEC_PATH), tp=4)
    nv = _predict_with_bandwidth("nvlink", 2.0)
    ib = _predict_with_bandwidth("ib", 2.0)

    def comm(r):
        return [c for c in r["components"] if c["kind"] == "comm"][0]["seconds"]

    for a, b in ((base, nv), (base, ib)):
        for c1, c2 in zip(a["components"], b["components"]):
            if c1["kind"] == "compute":
                assert c1["seconds"] == c2["seconds"], f"compute changed for {c1['op']}"

    assert comm(nv) < comm(base), "doubling NVLink must reduce intra-node TP comm"
    assert abs(comm(ib) - comm(base)) < 1e-15, (
        "doubling IB changed the TP comm term: the TP group is not intra-node")


def test_trace_has_predicted_true():
    result = cy0_l1.predict_layer(str(SPEC_PATH))
    for c in result["components"]:
        assert c.get("predicted") is True


def test_jsonl_emission():
    import subprocess
    out = subprocess.run(
        [sys.executable, str(REPO / "cy0_l1.py"), "--spec", str(SPEC_PATH)],
        capture_output=True, text=True
    )
    lines = [l for l in out.stdout.strip().split("\n") if l]
    assert len(lines) == len(cy0_l1.layer_ops())
    for line in lines:
        rec = json.loads(line)
        assert "kind" in rec
        assert "op" in rec
        assert "bytes" in rec
        assert "seconds" in rec


def _cross_node_spec():
    """Spec copy whose ranks 0 and 1 sit on DIFFERENT nodes, so the path is IB.

    After gpu-sim-by9 the rack spec places ranks contiguously per node, so ranks
    0 and 1 are intra-node and resolve over NVLink. A magnitude test for the IB
    path must say which ranks it uses rather than assume a placement.
    """
    import copy
    spec = copy.deepcopy(load_spec(str(SPEC_PATH)))
    spec["ranks"][1] = {"rank": 1, "node": "node1", "device": 0}
    return spec


def test_magnitude_known_link():
    """Magnitude test: known link, known transfer, hand-derived expected time.

    Two fabrics, both derived term by term from epic_cluster/link_model.py:
      payload S = 500000 float32 = 2,000,000 bytes
      ring n=2: bw factor 2(n-1)/n = 1.0 ; latency factor 2(n-1) = 2
      pieces  : 8 (explicit; n>=3 is refused, never cap-derived)

    NVLink (ranks 0-1 on one node, 900 GB/s, 1 us):
      bw 1.0*2e6/900e9 = 2.22 us ; lat 2*1us*8 = 16 us ; total 18.22 us
    IB     (ranks 0-1 on different nodes, 50 GB/s, 5 us):
      bw 1.0*2e6/50e9  = 40.0 us ; lat 2*5us*8 = 80 us ; total 120.0 us

    5% tolerance, not a decade window: this catches a units error (8x), a
    shared_group error (8x), a wrong-fabric error (18x here) or a factor-of-n
    error in the ring maths.
    """
    import epic_cluster.link_model as lm
    kw = dict(algorithm="ring_allreduce", count=500000, dtype="float32",
              n_ranks=2, pieces_per_phase=8)

    nvlink_us = lm.predict(load_spec(str(SPEC_PATH)), **kw)["duration_s"] * 1e6
    ib_us = lm.predict(_cross_node_spec(), **kw)["duration_s"] * 1e6

    for actual, expected, name in ((nvlink_us, 18.22, "NVLink"),
                                   (ib_us, 120.0, "IB")):
        assert abs(actual - expected) / expected < 0.05, (
            f"{name} magnitude mismatch: got {actual:.2f} us, "
            f"expected {expected:.2f} us")


def test_shared_group_costs_a_factor_of_eight():
    """The trap that produced a real 8x error, pinned as a test.

    An NDR node has EIGHT independent 400 Gb/s NICs, one per GPU, so shared_group
    is null and each GPU sees its own 50 GB/s. Putting all eight inter-node links
    in one shared_group divides the per-GPU bandwidth by 8 -- a different
    physical claim (one 50 GB/s trunk shared by eight GPUs).
    """
    import epic_cluster.link_model as lm
    kw = dict(algorithm="ring_allreduce", count=500000, dtype="float32",
              n_ranks=2, pieces_per_phase=8)

    independent = _cross_node_spec()
    shared = _cross_node_spec()
    for link in shared["links"]:
        if link["kind"] == "ib":
            link["shared_group"] = "ib"

    def us(spec):
        return lm.predict(spec, **kw)["duration_s"] * 1e6

    assert abs(us(independent) - 120.0) / 120.0 < 0.05, us(independent)
    # one shared trunk: bw 50/8 GB/s -> 320 us + lat 80 us = 400 us
    assert abs(us(shared) - 400.0) / 400.0 < 0.05, us(shared)
    assert us(shared) / us(independent) > 3.0, "sharing must cost a large factor"
