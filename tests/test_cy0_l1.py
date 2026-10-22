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


def test_ib_bandwidth_doubles_comm_not_compute():
    result1 = cy0_l1.predict_layer(str(SPEC_PATH))
    # Double IB bandwidth
    spec = json.loads(SPEC_PATH.read_text())
    for link in spec["links"]:
        if link["kind"] == "ib":
            link["bw_gbps"] *= 2
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        f.flush()
        result2 = cy0_l1.predict_layer(f.name)
        os.unlink(f.name)

    # Compute terms unchanged
    for c1, c2 in zip(result1["components"], result2["components"]):
        if c1["kind"] == "compute":
            assert c1["seconds"] == c2["seconds"], f"compute changed for {c1['op']}"

    # Comm term strictly reduced
    comm1 = [c for c in result1["components"] if c["kind"] == "comm"][0]
    comm2 = [c for c in result2["components"] if c["kind"] == "comm"][0]
    assert comm2["seconds"] < comm1["seconds"], "comm did not decrease"


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


def test_magnitude_known_link():
    """Magnitude test: one known link, one known transfer, hand-derived expected time.

    Derivation, all terms from epic_cluster/link_model.py and nothing else:
      link        : NDR IB, 50 GB/s, alpha = 5 us (latency_us)
      payload     : count=500000 float32 -> S = 2,000,000 bytes
      n_ranks     : 2   -> ring bw factor  2(n-1)/n = 1.0
                          ring latency factor 2(n-1) = 2
      pieces      : 8 (passed explicitly; n>=3 is refused, never cap-derived)
      bw_term     : 1.0 * 2e6 / 50e9 = 4.0e-5 s = 40 us
      lat_term    : 2.0 * 5e-6 * 8   = 8.0e-5 s = 80 us
      expected    : 120 us

    A 5% tolerance, not a decade window: this test exists to catch a units error
    (8x), a shared_group error (8x) or a factor-of-n error in the ring maths.
    """
    import epic_cluster.link_model as lm
    spec = load_spec(str(SPEC_PATH))
    res = lm.predict(spec, "ring_allreduce", 500000, "float32", n_ranks=2, pieces_per_phase=8)
    expected_us = 120.0
    actual_us = res["duration_s"] * 1e6
    assert abs(actual_us - expected_us) / expected_us < 0.05, \
        f"magnitude mismatch: got {actual_us} us, expected {expected_us} us"


def test_shared_group_costs_a_factor_of_eight():
    """The trap that produced a real 8x error, pinned as a test.

    An NDR node has EIGHT independent 400 Gb/s NICs, one per GPU, so each GPU
    sees its own 50 GB/s -- hence shared_group=null on the IB links. Putting all
    eight links in one shared_group makes link_model divide the per-GPU
    bandwidth by 8, which is a DIFFERENT physical claim (one 50 GB/s trunk
    shared by eight GPUs) and makes every cross-node term 8x too slow.
    This test asserts both branches, so a spec edit that reintroduces the
    sharing is visible as a value change rather than a silent slowdown.
    """
    import copy
    import epic_cluster.link_model as lm

    independent = load_spec(str(SPEC_PATH))
    shared = copy.deepcopy(independent)
    for link in shared["links"]:
        if link["kind"] == "ib":
            link["shared_group"] = "ib"

    def us(spec):
        return lm.predict(spec, "ring_allreduce", 500000, "float32",
                          n_ranks=2, pieces_per_phase=8)["duration_s"] * 1e6

    assert abs(us(independent) - 120.0) / 120.0 < 0.05
    # one shared trunk: bw 50/8 GB/s -> bw_term 320 us + lat 80 us = 400 us
    assert abs(us(shared) - 400.0) / 400.0 < 0.05
    assert us(shared) / us(independent) > 3.0, "sharing must cost a large factor"
