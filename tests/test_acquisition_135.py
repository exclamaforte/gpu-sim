#!/usr/bin/env python3
"""Tests for 13.5 (CPU): selector contracts + retrospective on REAL head
outputs (11.2 predictions, read cross-branch read-only) + loop gate."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.acquisition import (  # noqa: E402
    overlap_rate,
    stratified_topk_by_variance,
    topk_by_variance,
)

P112 = "lab/gpu-sim-u2o.11.2:experiments/gpu-sim-u2o.11.2/predictions.json"


def _preds():
    out = subprocess.run(["git", "show", P112], capture_output=True,
                         text=True, cwd=REPO)
    if out.returncode != 0:
        raise unittest.SkipTest("11.2 branch predictions unavailable")
    return json.loads(out.stdout)


class TestSelector(unittest.TestCase):
    def test_order_and_stratification(self):
        pool = [{"hash": f"r{i}", "op": "a" if i % 2 else "b", "var": float(i),
                 "sigma_log": 0.0} for i in range(10)]
        self.assertEqual([p["hash"] for p in topk_by_variance(pool, 3)],
                         ["r9", "r8", "r7"])
        strat = stratified_topk_by_variance(pool, 4)
        self.assertEqual(sorted(p["op"] for p in strat),
                         ["a", "a", "b", "b"])

    def test_overlap_metric(self):
        a = [{"hash": "x"}, {"hash": "y"}]
        self.assertEqual(overlap_rate(a, a), 1.0)
        self.assertEqual(overlap_rate([{"hash": "x"}], a), 0.5)


class TestRetrospective(unittest.TestCase):
    def test_variance_finds_failure_region(self):
        preds = _preds()
        self.assertEqual(len(preds), 57)
        worst = sorted(preds, key=lambda p: -p["abs_err"])[:10]
        sel = topk_by_variance(preds, 10)
        ov = overlap_rate(sel, worst)
        in_triple = sum(1 for p in sel
                        if p["triple"] == "latency->memory")
        metrics = {"bead": "gpu-sim-u2o.13.5",
                   "n_pool": len(preds),
                   "var_top10_overlap_with_err_top10": round(ov, 2),
                   "var_top10_in_failure_triple": f"{in_triple}/10",
                   "failure_triple_base_rate": "28/57",
                   "gate_open": False}
        (REPO / "experiments/gpu-sim-u2o.13.5"
         / "metrics.json").write_text(json.dumps(metrics, indent=2))
        # HONEST NEGATIVE: total-variance acquisition is ~chance on exact
        # rows (0.2 vs 0.175 random expectation) and triple enrichment
        # (7/10 vs 28/57 base, p~0.2) is not significant -- consistent with
        # 11.2's over-coverage (intervals too wide => var uninformative).
        # Pinned exactly: if the head improves, these move and the test
        # forces the interpretation to be rewritten, not silently kept.
        self.assertAlmostEqual(ov, 0.2)
        self.assertEqual(in_triple, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
