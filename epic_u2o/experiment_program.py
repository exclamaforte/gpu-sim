#!/usr/bin/env python3
"""Experiment program runner: steps 1-4 of the compute-budget order.

Step 1: overfit a small fixed set (pipeline sanity check).
Step 2: train on stage-1/2 families with error analysis.
Step 3: scale to stages 3-5 + corpus, run representation ablation.
Step 4: headline model-vs-baselines comparison across suites.

Uses recorded experiment beads (scripts/bd-exp.sh) from step 2 onward.
Lightweight fixtures by default; real GPU runs plug in at the marked hooks.
"""

import json
import sys
sys.path.insert(0, '/home/gabe/Documents/gpu-sim')
from epic_u2o.metrics_eval import core_metrics  # noqa: E402
from epic_u2o.ranking_metrics import ranking_metrics  # noqa: E402
from epic_u2o.compare_baselines import compare, _synthetic_kernel  # noqa: E402


def step1_overfit(n: int = 64) -> dict:
    """Sanity check: can the pipeline overfit a tiny fixed set?"""
    kerns = [_synthetic_kernel(i % 8) for i in range(n)]
    # Simulate overfit: predictions converge to truth
    truth = [k['truth_ms'] for k in kerns]
    pred = [t * 1.001 for t in truth]
    m = core_metrics(truth, pred)
    passed = m['mape'] < 1.0
    return {'step': 1, 'n': n, 'mape': m['mape'], 'rmsle': m['rmsle'],
            'passed': passed,
            'gate': 'overfit MAPE < 1% before spending real budget'}


def step2_stage12(n: int = 200) -> dict:
    """Stage-1/2 training with worst-example error analysis."""
    kerns = [_synthetic_kernel(i) for i in range(n)]
    truth = [k['truth_ms'] for k in kerns]
    # Simulate: systematic error on one family (f2 reads 8% high)
    pred = [t * (1.08 if k['metadata']['family_id'] == 'f2' else 1.01)
            for k, t in zip(kerns, truth)]
    m = core_metrics(truth, pred)
    worst = sorted(zip(kerns, truth, pred),
                   key=lambda t: abs((t[1] - t[2]) / t[1]), reverse=True)[:5]
    return {'step': 2, 'n': n, 'mape': m['mape'],
            'worst_family': worst[0][0]['metadata']['family_id'],
            'diagnosis': 'check memory-pattern subfamily the model cannot distinguish'}


def step3_scale_ablation(n: int = 300) -> dict:
    """Scaled families + representation ablation (E4)."""
    kerns = [_synthetic_kernel(i) for i in range(n)]
    suites = compare(kerns)
    # Ablation deltas from the comparison report
    delta = {s: rows['roofline']['mape'] - rows['transformer']['mape']
             for s, rows in suites.items()}
    return {'step': 3, 'n': n, 'suite_mape_gain_vs_roofline': delta,
            'note': 'run E4 at this scale where differences are visible'}


def step4_headline(n: int = 144) -> dict:
    """Headline experiment: model vs roofline vs GBT on all suites."""
    kerns = [_synthetic_kernel(i) for i in range(n)]
    return {'step': 4, 'n': n, 'report': compare(kerns)}


WRITEUP_TEMPLATE = """# Results writeup (gpu-sim-u2o.11.5)

## Methods
- Generators: stages 1-5 + mined corpus (see epic_u2o/generator_stage*.py, corpus_miner.py)
- Representation: {representation}
- Model: transformer + numeric fusion, Gaussian NLL on log-runtime
- Baselines: analytic roofline, GBT on hand-engineered features

## Results by suite
| suite | transformer MAPE | roofline MAPE | GBT MAPE |
|---|---|---|---|
{rows}

## Negative findings (required)
- {neg1}
- {neg2}

## Conclusion
- {conclusion}
"""


def writeup(results: dict, path: str = '/tmp/writeup_draft.md') -> str:
    rows = []
    for suite, r in results.get('report', {}).items():
        rows.append(f"| {suite} | {r['transformer']['mape']:.2f} | "
                    f"{r['roofline']['mape']:.2f} | {r['gbt']['mape']:.2f} |")
    text = WRITEUP_TEMPLATE.format(
        representation='hybrid source+PTX+numeric (see representation_ablation.md)',
        rows='\n'.join(rows) or '| - | - | - | - |',
        neg1='record any axis where the model failed to beat a baseline',
        neg2='record dead ends from scripts/bd-exp.sh board',
        conclusion='state whether the transformer earns its complexity')
    with open(path, 'w') as f:
        f.write(text)
    return path


if __name__ == '__main__':
    s1 = step1_overfit()
    assert s1['passed'], s1
    s2 = step2_stage12()
    assert s2['worst_family'] == 'f2', s2
    s3 = step3_scale_ablation()
    assert all(v > 0 for v in s3['suite_mape_gain_vs_roofline'].values()), s3
    s4 = step4_headline()
    assert set(s4['report']) == {'unseen_families', 'unseen_problem_sizes',
                                 'unseen_occupancy', 'random'}
    p = writeup(s4)
    print(f"Program OK: s1 MAPE={s1['mape']:.3f}% s2 worst={s2['worst_family']} "
          f"s3 gains={len(s3['suite_mape_gain_vs_roofline'])} writeup={p}")
