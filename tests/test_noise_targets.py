"""gpu-sim-u2o.6.6: the targets artifact holds what the design promises.

Noise per cell (never pooled), bias beside it (never subtracted), regimes on
every row, the estimator proven on 6.6.1's known-differing cell. CPU-only.
"""
import importlib.util
import json
import math
import statistics
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "gpu-sim-u2o.6.6"

spec = importlib.util.spec_from_file_location(
    "build_targets_mod", EXP / "build_targets.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def corpus():
    rows = []
    for c in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl"):
        rows += [json.loads(l) for l in (REPO / c).read_text().splitlines()]
    return {r["kernel_hash"]: r for r in rows}


def targets():
    p = EXP / "targets.jsonl"
    if not p.exists():
        raise unittest.SkipTest("run build_targets.py first")
    return [json.loads(l) for l in p.read_text().splitlines()]


def metrics():
    p = EXP / "metrics.json"
    if not p.exists():
        raise unittest.SkipTest("run build_targets.py first")
    return json.loads(p.read_text())


class TestEstimator(unittest.TestCase):
    def test_median_of_differing_invocations(self):
        # Item 9: 6.6.1's known cell spanned 1.472 / 2.208 / 2.304 us.
        self.assertEqual(bt.cell_median([1.472, 2.208, 2.304]), 2.208)

    def test_estimator_is_median_not_mean(self):
        meds = [1.472, 2.208, 2.304]
        self.assertNotAlmostEqual(bt.cell_median(meds),
                                  statistics.fmean(meds), places=3)
        self.assertEqual(len(meds), 3)  # n rides along, per item 2

    def test_two_invocation_median_is_their_mean(self):
        # Corpus grain (n=2): median == mean, identity not assumption.
        self.assertEqual(bt.cell_median([1.0, 3.0]), 2.0)


class TestSigma(unittest.TestCase):
    def test_zero_cv_maps_to_zero_exactly(self):
        self.assertEqual(bt.sigma_log_from_cv(0.0), 0.0)

    def test_lognormal_map_at_unit_cv(self):
        self.assertAlmostEqual(bt.sigma_log_from_cv(1.0),
                               math.sqrt(math.log(2.0)))

    def test_monotone_in_cv(self):
        cvs = [0.0, 0.001, 0.01, 0.1, 0.5]
        sigs = [bt.sigma_log_from_cv(c) for c in cvs]
        self.assertEqual(sigs, sorted(sigs))
        self.assertGreater(sigs[-1], 0)

    def test_sigma_uses_only_the_cell_own_cv(self):
        # No pooling: recompute one row by hand from its own ai_cv alone.
        rows, tgts = corpus(), {t["kernel_hash"]: t for t in targets()}
        h = "66a4b75ee7ee19a8"  # scale n=2^22 exemplar
        self.assertAlmostEqual(
            tgts[h]["sigma_log"],
            math.sqrt(math.log1p(rows[h]["ai_cv"] ** 2)))


class TestTargetRows(unittest.TestCase):
    def test_population(self):
        self.assertEqual(len(targets()), 456)

    def test_item8_fields_on_every_row(self):
        need = {"median_runtime_ms", "sigma_log", "n_invocation",
                "measured_regime", "roofline_regime", "regime_disagree",
                "roofline_triple", "model_bias", "protocol",
                "targets_version"}
        for t in targets():
            self.assertTrue(need <= set(t), t["kernel_hash"])

    def test_bias_never_subtracted(self):
        # Item 7 structural: target median IS the committed row median.
        rows = corpus()
        for t in targets():
            self.assertEqual(t["median_runtime_ms"],
                             rows[t["kernel_hash"]]["median_runtime_ms"],
                             t["kernel_hash"])

    def test_bias_keyed_by_triple(self):
        expect = {"latency->latency": -0.5638, "latency->memory": 0.378,
                  "memory->memory": 1.1164}
        for t in targets():
            self.assertEqual(t["model_bias"], expect[t["roofline_triple"]],
                             t["kernel_hash"])

    def test_protocols_split(self):
        protos = [t["protocol"] for t in targets()]
        self.assertEqual(protos.count("single_run_7_1"), 264)
        self.assertEqual(protos.count("single_run_e5q"), 192)

    def test_disagreement_flags(self):
        flags = [t for t in targets() if t["regime_disagree"]]
        self.assertEqual(len(flags), 43)
        for f in flags:
            self.assertEqual(f["measured_regime"], "launch_bound")
            self.assertEqual(f["roofline_regime"], "memory")

    def test_zero_sigma_count(self):
        self.assertEqual(sum(1 for t in targets()
                             if t["sigma_log"] == 0.0), 2)

    def test_poor_flag_carried(self):
        poor = [t for t in targets() if t["deliberately_poor"]]
        self.assertEqual(len(poor), 8)


class TestNoiseCharacterised(unittest.TestCase):
    def test_latency_noise_dwarfs_memory_noise(self):
        m = metrics()["noise_by_roofline_triple"]
        lat = m["latency->latency"]["median_ai_cv"]
        mem = m["memory->memory"]["median_ai_cv"]
        self.assertGreater(lat, 5 * mem)

    def test_per_regime_n_attached(self):
        m = metrics()["noise_by_roofline_triple"]
        self.assertEqual(
            {k: v["n"] for k, v in m.items()},
            {"latency->latency": 220, "latency->memory": 204,
             "memory->memory": 32})

    def test_contention_clean(self):
        c = metrics()["contention"]
        self.assertEqual(c["n_contended"], 0)
        self.assertEqual(c["n_excluded"], 0)


if __name__ == "__main__":
    unittest.main()
