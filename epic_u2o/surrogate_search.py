#!/usr/bin/env python3
"""Launch-config surrogate search (13.3). The SEARCH machinery, model-free:
rank candidate configs by predicted mean (greedy) or UCB, report regret
against measured truth when available. Validated on synthetic configs with
known optima; REAL multi-config kernels don't exist in the corpus yet
(one config per kernel_hash everywhere -- gate test pins this), so the
GPU-validation half waits for a measurement campaign + a cost model that
beats GBT."""
import math


def rank_configs(predict, configs, k=3, ucb_beta=0.0):
    """predict(cfg) -> (mu, var). Lower mu is better (latency).

    ucb_beta=0: greedy on mu. ucb_beta>0: mu - beta*std (optimistic).
    Returns [(cfg, mu, var)] sorted best-first, top-k.
    """
    scored = []
    for c in configs:
        mu, var = predict(c)
        score = mu - ucb_beta * math.sqrt(max(var, 0.0))
        scored.append((score, c, mu, var))
    scored.sort(key=lambda t: t[0])
    return [(c, mu, var) for _, c, mu, var in scored[:k]]


def regret(selected, truth):
    """selected: list of configs (best-first). truth: cfg->measured ms.
    Returns (best measured among selected) - (global best). 0 is optimal.
    """
    best_global = min(truth.values())
    best_sel = min(truth[c] for c, _, _ in selected)
    return best_sel - best_global


if __name__ == "__main__":
    truth = {f"cfg{i}": 10.0 + abs(i - 7) * 1.5 for i in range(16)}
    perfect = lambda c: (truth[c], 0.01)  # noqa: E731
    top = rank_configs(perfect, list(truth), k=3)
    assert regret(top, truth) == 0.0 and top[0][0] == "cfg7"
    print("surrogate_search OK: perfect surrogate recovers cfg7, regret 0")
