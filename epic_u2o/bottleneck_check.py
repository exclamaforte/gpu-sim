#!/usr/bin/env python3
"""Bottleneck checkpoint: data-generation vs training throughput.

Explicit gate before investing in training-speed optimization: compare
measured data-generation throughput (epic C7 / throughput_tracker) against
measured training throughput (12.1 profile). Only open further
training-optimization work if training dominates wall-clock time.
"""

import sys
sys.path.insert(0, '/home/gabe/Documents/gpu-sim')
from epic_u2o.training_throughput_profile import profile as train_profile  # noqa: E402


def check(data_gen_ex_per_hour: float = 1200.0,
          train_examples_needed: int = 100000) -> dict:
    """Compare wall-clock for a fixed data budget.

    Defaults: data-gen 1200 ex/hr is a placeholder; replace with the measured
    value from runs/logs/throughput_*.jsonl once the GPU pipeline runs.
    """
    t = train_profile(n_steps=3)
    # Training examples/sec from step time: batch / step_time
    train_ex_per_sec = t['batch'] / t['step_time_s'] if t['step_time_s'] > 0 else 0.0
    train_ex_per_hour = train_ex_per_sec * 3600.0
    data_gen_hours = train_examples_needed / data_gen_ex_per_hour
    train_hours = train_examples_needed / train_ex_per_hour if train_ex_per_hour > 0 else float('inf')
    bottleneck = 'data-generation' if data_gen_hours >= train_hours else 'training'
    decision = ('INVEST in training speed' if bottleneck == 'training'
                else 'DO NOT invest in training speed; optimize data generation first')
    return {'data_gen_ex_per_hour': data_gen_ex_per_hour,
            'train_ex_per_hour_cpu_smoke': train_ex_per_hour,
            'budget_examples': train_examples_needed,
            'data_gen_hours': data_gen_hours,
            'train_hours_cpu_smoke': train_hours,
            'bottleneck': bottleneck, 'decision': decision,
            'caveat': 'train rows are CPU-smoke scale; rerun both sides on the 5090'}


if __name__ == '__main__':
    r = check()
    assert r['bottleneck'] == 'data-generation', r
    print(f"Bottleneck OK: {r['bottleneck']} "
          f"(data-gen {r['data_gen_hours']:.1f}h vs train-smoke {r['train_hours_cpu_smoke']:.4f}h) -> {r['decision']}")
