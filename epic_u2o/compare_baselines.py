#!/usr/bin/env python3
"""Compare transformer vs roofline vs GBT across every suite (single source of truth).

Runs all three predictors through core + ranking metrics on each
generalization suite. Uses lightweight synthetic fixtures by default;
swap in real measurements when the GPU pipeline is available.
"""

import json
import sys
sys.path.insert(0, '/home/gabe/Documents/gpu-sim')
from epic_u2o.metrics_eval import core_metrics  # noqa: E402
from epic_u2o.ranking_metrics import ranking_metrics  # noqa: E402
from epic_u2o.generalization_suites import build_suites  # noqa: E402


def _synthetic_kernel(i: int) -> dict:
    fams = [('f1', 'scale'), ('f2', 'reduce_sum_block'), ('f3', 'coalesced_read')]
    fam, op = fams[i % 3]
    n = (1024, 8192, 65536, 262144)[(i // 3) % 4]
    blk = (64, 256, 1024)[(i // 12) % 3]
    # Ground truth: simple size-driven time with family offset
    truth = 0.01 + n * 1e-6 * (1.0 + 0.2 * (i % 3))
    return {'op_type': op, 'problem_size': n, 'block': [blk, 1, 1],
            'grid': [8, 1, 1], 'source': 'x',
            'metadata': {'family_id': fam, 'op_type': op, 'generator': 'test'},
            'truth_ms': truth}


def _predict(method: str, kernels: list) -> list:
    """Deterministic stub predictors with known ordering: transformer best."""
    preds = []
    for i, k in enumerate(kernels):
        t = k['truth_ms']
        if method == 'transformer':
            preds.append(t * (1.0 + 0.02 * ((i % 5) - 2)))
        elif method == 'roofline':
            preds.append(t * (1.0 + 0.10 * ((i % 5) - 2)))
        else:  # gbt
            preds.append(t * (1.0 + 0.06 * ((i % 7) - 3)))
    return preds


def compare(kernels: list) -> dict:
    suites = build_suites(kernels)
    report = {}
    for suite_name in ('unseen_families', 'unseen_problem_sizes', 'unseen_occupancy', 'random'):
        s = suites[suite_name]
        test = s['test'] or s['train'][:8]
        truth = [k['truth_ms'] for k in test]
        rows = {}
        for method in ('transformer', 'roofline', 'gbt'):
            pred = _predict(method, test)
            rows[method] = {**core_metrics(truth, pred),
                            **ranking_metrics(truth, pred)}
        report[suite_name] = rows
    return report


if __name__ == '__main__':
    kerns = [_synthetic_kernel(i) for i in range(36)]
    rep = compare(kerns)
    # Transformer stub must win on MAPE in every suite
    for suite, rows in rep.items():
        assert rows['transformer']['mape'] < rows['roofline']['mape'], suite
        assert rows['transformer']['spearman'] >= rows['roofline']['spearman'] - 0.05, suite
    print("Compare OK")
    print(json.dumps({s: {m: round(r['mape'], 2) for m, r in rows.items()}
                      for s, rows in rep.items()}, indent=1))
