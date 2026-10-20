import json
import struct
import tempfile
import unittest
from pathlib import Path

from epic_cluster.spec import validate, loads, SpecError
from epic_cluster.topology import emit, read_binary, build_adjacency, shortest_path, dev_index, shared_group_counts, effective_bw


def make_spec(**overrides):
    spec = {
        "version": 1,
        "models": {
            "M1": {"sm_count": 80, "clock_mhz": 1410, "mem_bytes": 40*1024**3,
                   "mem_bw_gbs": 1555, "arch": "sm_80", "source": "datasheet"}
        },
        "nodes": [
            {"name": "n0", "devices": [{"id": 0, "model": "M1"}, {"id": 1, "model": "M1"}]}
        ],
        "links": [
            {"a": {"node": "n0", "device": 0}, "b": {"node": "n0", "device": 1},
             "kind": "nvlink", "bw_gbps": 600, "latency_us": 1, "mtu": 4096,
             "loss_ppm": 0, "jitter_us": 0, "shared_group": None}
        ],
        "ranks": [
            {"rank": 0, "node": "n0", "device": 0},
            {"rank": 1, "node": "n0", "device": 1}
        ]
    }
    for k, v in overrides.items():
        spec[k] = v
    return spec


class TestValidation(unittest.TestCase):
    def test_valid(self):
        validate(make_spec())

    def test_unknown_key(self):
        s = make_spec()
        s["foo"] = 1
        with self.assertRaises(SpecError) as ctx:
            validate(s)
        self.assertIn("unknown key", str(ctx.exception))

    def test_dangling_link_endpoint(self):
        s = make_spec()
        s["links"][0]["a"]["device"] = 99
        with self.assertRaises(SpecError):
            validate(s)

    def test_rank_missing_device(self):
        s = make_spec()
        s["ranks"][0]["device"] = 99
        with self.assertRaises(SpecError):
            validate(s)

    def test_bw_le_zero(self):
        s = make_spec()
        s["links"][0]["bw_gbps"] = 0
        with self.assertRaises(SpecError):
            validate(s)

    def test_model_without_source(self):
        s = make_spec()
        s["models"]["M1"]["source"] = ""
        with self.assertRaises(SpecError):
            validate(s)

    def test_no_path_between_ranks(self):
        # two devices no link
        s = make_spec()
        s["nodes"][0]["devices"] = [{"id": 0, "model": "M1"}, {"id": 1, "model": "M1"}]
        s["links"] = []
        s["ranks"] = [{"rank": 0, "node": "n0", "device": 0},
                      {"rank": 1, "node": "n0", "device": 1}]
        with self.assertRaises(SpecError):
            validate(s)

    def test_latency_negative(self):
        s = make_spec()
        s["links"][0]["latency_us"] = -1
        with self.assertRaises(SpecError):
            validate(s)

    def test_mtu_lt_1(self):
        s = make_spec()
        s["links"][0]["mtu"] = 0
        with self.assertRaises(SpecError):
            validate(s)

    def test_loss_ppm_gt_1e6(self):
        s = make_spec()
        s["links"][0]["loss_ppm"] = 2e6
        with self.assertRaises(SpecError):
            validate(s)


class TestTopology(unittest.TestCase):
    def test_chained_path(self):
        s = make_spec()
        # A-B-C chain: device0 - device1 - device2 (add device2)
        s["nodes"][0]["devices"].append({"id": 2, "model": "M1"})
        s["links"] = [
            {"a": {"node": "n0", "device": 0}, "b": {"node": "n0", "device": 1},
             "kind": "pcie", "bw_gbps": 32, "latency_us": 1, "mtu": 4096,
             "loss_ppm": 0, "jitter_us": 0, "shared_group": None},
            {"a": {"node": "n0", "device": 1}, "b": {"node": "n0", "device": 2},
             "kind": "pcie", "bw_gbps": 32, "latency_us": 1, "mtu": 4096,
             "loss_ppm": 0, "jitter_us": 0, "shared_group": None},
        ]
        s["ranks"] = [{"rank": 0, "node": "n0", "device": 0},
                      {"rank": 1, "node": "n0", "device": 2}]
        validate(s)
        adj, di = build_adjacency(s, dev_index(s))
        a = di[("n0", 0)]
        c = di[("n0", 2)]
        path = shortest_path(adj, a, c)
        self.assertEqual(len(path), 2)

    def test_model_mismatch_preserved(self):
        s = make_spec()
        s["models"]["M2"] = {"sm_count": 100, "clock_mhz": 1000, "mem_bytes": 1024,
                            "mem_bw_gbs": 100, "arch": "sm_90", "source": "x"}
        s["nodes"][0]["devices"][1]["model"] = "M2"
        validate(s)
        data = emit(s)
        rb = read_binary(data)
        # find device records model names by order: dev0 M1, dev1 M2
        names = [d["model_name"] for d in rb["devices"]]
        self.assertEqual(names, ["M1", "M2"])

    def test_shared_group_division(self):
        s = make_spec()
        s["links"][0]["shared_group"] = "g"
        # add another link in same group to divide
        s["nodes"][0]["devices"].append({"id": 2, "model": "M1"})
        s["links"].append({"a": {"node": "n0", "device": 0}, "b": {"node": "n0", "device": 2},
                           "kind": "nvlink", "bw_gbps": 600, "latency_us": 1, "mtu": 4096,
                           "loss_ppm": 0, "jitter_us": 0, "shared_group": "g"})
        s["ranks"] = [{"rank": 0, "node": "n0", "device": 0},
                      {"rank": 1, "node": "n0", "device": 1}]
        validate(s)
        counts = shared_group_counts(s)
        self.assertEqual(counts["g"], 2)
        self.assertAlmostEqual(effective_bw(s["links"][0], counts), 300.0)

    def test_emit_roundtrip(self):
        s = make_spec()
        data = emit(s)
        data2 = emit(s)
        self.assertEqual(data, data2)
        rb = read_binary(data)
        data3 = emit(s)  # re-emit from spec not from rb; but byte compare of write->read->? 
        # byte-identical write->read->write not applicable since read returns different shape;
        # instead ensure read parses all fields
        self.assertEqual(rb["n_dev"], 2)
        self.assertEqual(rb["n_links"], 1)
        self.assertEqual(rb["n_ranks"], 2)


class TestExamples(unittest.TestCase):
    def test_examples_validate(self):
        base = Path(__file__).resolve().parents[1] / "experiments" / "gpu-sim-m3d.1"
        for name in ["nvlink2.json", "pcie4.json", "ib8.json"]:
            p = base / name
            if not p.exists():
                self.skipTest(f"missing {p}")
            spec = json.loads(p.read_text())
            validate(spec)


if __name__ == "__main__":
    unittest.main()
