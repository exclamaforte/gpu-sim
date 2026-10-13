#!/usr/bin/env python3
"""Tests for 13.3 (CPU): search machinery on synthetic configs + corpus
gate (no multi-config kernels measured anywhere)."""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.surrogate_search import rank_configs, regret  # noqa: E402


class TestSearch(unittest.TestCase):
    def test_perfect_surrogate_zero_regret(self):
        truth = {f"c{i}": 10.0 + abs(i - 7) * 1.5 for i in range(16)}
        top = rank_configs(lambda c: (truth[c], 0.01), list(truth), k=3)
        self.assertEqual(top[0][0], "c7")
        self.assertEqual(regret(top, truth), 0.0)

    def test_noisy_surrogate_bounded_regret(self):
        truth = {f"c{i}": 10.0 + abs(i - 7) * 1.5 for i in range(16)}
        biased = {c: v + (2.0 if i % 2 else -2.0)
                  for i, (c, v) in enumerate(truth.items())}
        top = rank_configs(lambda c: (biased[c], 0.5), list(truth), k=3)
        self.assertLessEqual(regret(top, truth), 3.0)

    def test_ucb_explores_uncertain_arm(self):
        # greedy picks the sure mediocre arm; UCB tries the uncertain one
        def predict(c):
            return {"sure": (5.0, 0.01), "wild": (6.0, 9.0)}[c]
        greedy = rank_configs(predict, ["sure", "wild"], k=1,
                              ucb_beta=0.0)
        ucb = rank_configs(predict, ["sure", "wild"], k=1, ucb_beta=2.0)
        self.assertEqual(greedy[0][0], "sure")
        self.assertEqual(ucb[0][0], "wild")

    def test_multi_config_gate_closed(self):
        import glob
        max_cfgs, n_files = 0, 0
        for f in glob.glob(str(REPO / "experiments/gpu-sim-*/measured.jsonl")) + \
                   glob.glob(str(REPO / "experiments/gpu-sim-*/batch_rows.jsonl")) + \
                   glob.glob(str(REPO / "experiments/gpu-sim-u2o.6.7.1/batch_rows.jsonl")):
            n_files += 1
            byk = Counter(json.loads(l).get("kernel_hash", "?")
                          for l in open(f))
            if byk:
                max_cfgs = max(max_cfgs, max(byk.values()))
        self.assertGreater(n_files, 0)
        metrics = {"bead": "gpu-sim-u2o.13.3",
                   "measured_files_scanned": n_files,
                   "max_configs_per_kernel": max_cfgs,
                   "gate_open": False}
        (REPO / "experiments/gpu-sim-u2o.13.3"
         / "metrics.json").write_text(json.dumps(metrics, indent=2))
        self.assertLess(max_cfgs, 2, "gate opened: multi-config data exists")


if __name__ == "__main__":
    unittest.main(verbosity=2)
