#!/usr/bin/env python3
"""Generalization suites: unseen sizes, families, occupancy regimes.

Held-out suites analogous to cclm's generalization distributions, built on
the structural-family split design (gpu-sim-u2o.7.7): extrapolate to larger
N, hold out entire families, hold out occupancy/launch-config regimes.
"""

import json
import sys
sys.path.insert(0, '/home/gabe/Documents/gpu-sim')
from epic_u2o.structural_split import split_by_family  # noqa: E402


def _problem_size(k: dict) -> int:
    return int(k.get('problem_size', 0))


def _occupancy_proxy(k: dict) -> float:
    """Cheap occupancy proxy from block size: threads/block / 1024."""
    b = k.get('block', [1, 1, 1])
    t = (b[0] if len(b) > 0 else 1) * (b[1] if len(b) > 1 else 1) * (b[2] if len(b) > 2 else 1)
    return min(1.0, t / 1024.0)


def build_suites(kernels: list, seed: int = 42) -> dict:
    """Build the three generalization suites plus a random baseline split."""
    kernels = list(kernels)
    fams = sorted({k.get('metadata', {}).get('family_id') or k.get('op_type', '?') for k in kernels})
    holdout_fams = set(fams[len(fams) * 4 // 5:] or fams[-1:]) if fams else set()

    sizes = sorted({_problem_size(k) for k in kernels})
    size_cut = sizes[int(len(sizes) * 0.8)] if len(sizes) > 1 else (sizes[0] if sizes else 0)

    occ = sorted({_occupancy_proxy(k) for k in kernels})
    occ_cut = occ[int(len(occ) * 0.8)] if len(occ) > 1 else 1.0

    unseen_families = split_by_family(kernels, holdout_families=sorted(holdout_fams), seed=seed)
    size_train = [k for k in kernels if _problem_size(k) < size_cut]
    size_test = [k for k in kernels if _problem_size(k) >= size_cut]
    occ_train = [k for k in kernels if _occupancy_proxy(k) < occ_cut]
    occ_test = [k for k in kernels if _occupancy_proxy(k) >= occ_cut]
    random_split = split_by_family(kernels, seed=seed)
    return {
        'unseen_families': unseen_families,
        'unseen_problem_sizes': {'train': size_train, 'val': [], 'test': size_test,
                                 'cut': size_cut},
        'unseen_occupancy': {'train': occ_train, 'val': [], 'test': occ_test,
                             'cut': occ_cut},
        'random': random_split,
        'holdout_families': sorted(holdout_fams),
    }


def suite_report(suites: dict) -> dict:
    rep = {}
    for name in ('unseen_families', 'unseen_problem_sizes', 'unseen_occupancy', 'random'):
        s = suites[name]
        rep[name] = {k: (len(v) if isinstance(v, list) else v)
                     for k, v in s.items() if k in ('train', 'val', 'test', 'cut')}
    rep['holdout_families'] = suites['holdout_families']
    return rep


if __name__ == '__main__':
    # Deterministic fixture: 3 families x sizes x occupancy levels
    kerns = []
    for fam, op in [('f1', 'scale'), ('f2', 'reduce_sum_block'), ('f3', 'coalesced_read')]:
        for n in (1024, 8192, 65536, 262144):
            for blk in (64, 256, 1024):
                kerns.append({'op_type': op, 'problem_size': n, 'block': [blk, 1, 1],
                              'grid': [8, 1, 1], 'source': 'x',
                              'metadata': {'family_id': fam, 'op_type': op,
                                           'generator': 'test'}})
    suites = build_suites(kerns)
    rep = suite_report(suites)
    assert rep['unseen_problem_sizes']['test'] > 0 and rep['unseen_problem_sizes']['train'] > 0
    assert rep['unseen_occupancy']['test'] > 0
    assert rep['unseen_families']['test'] > 0
    # Family holdout must not leak into train
    train_fams = {k['metadata']['family_id'] for k in suites['unseen_families']['train']}
    test_fams = {k['metadata']['family_id'] for k in suites['unseen_families']['test']}
    assert not (train_fams & test_fams), "family leakage"
    print(f"Suites OK: {json.dumps(rep)}")
