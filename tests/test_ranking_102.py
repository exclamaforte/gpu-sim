#!/usr/bin/env python3
"""Tests for 10.2 ranking metrics (CPU). Verifies the implementations
against scipy on random data (with ties), pins hand-computed tie cases,
and runs every metric on REAL rows (per-op train-median baseline on the
family_holdout test split -- the audit's 'never run on real data' point).
"""
import json
import random
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))

from scipy.stats import kendalltau, spearmanr  # noqa: E402

from epic_u2o.ranking_metrics import (  # noqa: E402
    kendall_tau,
    ranking_metrics,
    spearman,
    topk_hit_rate,
)
import train_trans_91 as T  # noqa: E402

EXP = REPO / "experiments" / "gpu-sim-u2o.10.2"


class TestAgainstScipy(unittest.TestCase):
    def test_random_with_ties(self):
        rng = random.Random(7)
        for trial in range(25):
            n = rng.randint(4, 30)
            vals = [round(rng.gauss(0, 1), 1) for _ in range(n)]  # ties
            yt = [rng.choice(vals) for _ in range(n)]
            yp = [rng.choice(vals) for _ in range(n)]
            if len(set(yt)) < 2 or len(set(yp)) < 2:
                continue
            self.assertAlmostEqual(spearman(yt, yp),
                                   spearmanr(yt, yp).statistic, places=9)
            self.assertAlmostEqual(kendall_tau(yt, yp),
                                   kendalltau(yt, yp).statistic, places=9)

    def test_hand_tie_case(self):
        # conc=1, t_x=1, t_y=1 -> tau = 1/2; ranks give rho = 0.5 (see REPORT)
        self.assertAlmostEqual(kendall_tau([1, 1, 2], [1, 2, 2]), 0.5)
        self.assertAlmostEqual(spearman([1, 1, 2], [1, 2, 2]), 0.5)

    def test_extremes(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(kendall_tau([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(topk_hit_rate([1, 2, 3, 4], [1, 2, 3, 4], k=2), 1.0)
        self.assertEqual(topk_hit_rate([1, 2, 3, 4], [4, 3, 2, 1], k=2), 0.0)


class TestRealData(unittest.TestCase):
    def test_per_op_median_baseline_on_test_split(self):
        rows = [json.loads(l) for l in
                (REPO / "experiments/gpu-sim-u2o.9.1"
                 / "train_data_91.jsonl").read_text().splitlines()]
        parts, _, _ = T.family_holdout_split(rows, REPO)
        med = {}
        for r in parts["train"]:
            med.setdefault(r["op"], []).append(r["y"])
        med = {k: sorted(v)[len(v) // 2] for k, v in med.items()}
        gmean = sorted(r["y"] for r in parts["train"])[len(parts["train"]) // 2]
        te = parts["test"]
        yt = [r["y"] for r in te]
        yp = [med.get(r["op"], gmean) for r in te]
        m = ranking_metrics(yt, yp, k=5)
        # degenerate on purpose: the family_holdout test split is single-op
        # (relu), so any op-conditioned predictor is constant here -- this
        # proves the all-tie guards return 0.0 instead of NaN on real data.
        self.assertEqual(m["spearman"], 0.0)
        self.assertEqual(m["kendall_tau"], 0.0)
        # genuine signal: raw numeric feature 17 ranks relu runtimes
        yp2 = [r["num33"][17] for r in te]
        m2 = ranking_metrics(yt, yp2, k=5)
        self.assertGreater(m2["spearman"], 0.5)
        metrics = {"bead": "gpu-sim-u2o.10.2",
                   "n_test": len(te),
                   "tie_guard": ("per-op train-median (constant on "
                                   "single-op test split)",
                                   {k: round(v, 4)
                                    for k, v in m.items() if k != "n"}),
                   "feat17_ranker": {k: round(v, 4)
                                       for k, v in m2.items() if k != "n"},
                   "n": m2["n"]}
        (EXP / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("\n10.2 metrics:", json.dumps(metrics))


if __name__ == "__main__":
    unittest.main(verbosity=2)
