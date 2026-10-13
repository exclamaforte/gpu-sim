#!/usr/bin/env python3
"""Uncertainty acquisition for measurement prioritization (13.5).

Rank pool rows by predicted variance (+ sigma floor in quadrature),
with an op-stratified variant against mode collapse. The CLOSED LOOP
(measure selected, retrain) needs a GPU campaign and is gated; what ships
is the tested selector plus a retrospective on real head outputs.
"""
import math


def total_var(p):
    return p["var"] + p.get("sigma_log", 0.0) ** 2


def topk_by_variance(rows, k):
    """Highest total-variance rows first (pure exploitation of uncertainty)."""
    return sorted(rows, key=lambda p: -total_var(p))[:k]


def stratified_topk_by_variance(rows, k, key="op"):
    """Round-robin top-variance per group: coverage against collapse."""
    groups = {}
    for p in rows:
        groups.setdefault(p.get(key), []).append(p)
    for v in groups.values():
        v.sort(key=lambda p: -total_var(p))
    out, i = [], 0
    while len(out) < k and any(len(v) > i for v in groups.values()):
        for g in sorted(groups, key=str):
            if len(groups[g]) > i and len(out) < k:
                out.append(groups[g][i])
        i += 1
    return out


def overlap_rate(selected, reference):
    """Fraction of reference rows present in selected (by hash)."""
    ref = {p["hash"] for p in reference}
    sel = {p["hash"] for p in selected}
    return len(ref & sel) / len(ref) if ref else 0.0


if __name__ == "__main__":
    pool = [{"hash": f"r{i}", "op": "a" if i % 2 else "b",
             "var": float(i), "sigma_log": 0.0} for i in range(10)]
    top = topk_by_variance(pool, 3)
    assert [p["hash"] for p in top] == ["r9", "r8", "r7"]
    strat = stratified_topk_by_variance(pool, 4)
    assert sorted(p["op"] for p in strat) == ["a", "a", "b", "b"]
    print("acquisition OK: variance order + stratification hold")
