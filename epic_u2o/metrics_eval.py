#!/usr/bin/env python3
"""Core metrics: MAPE, log-scale RMSE, uncertainty calibration.

Relative-error metrics for heavy-tailed runtime distributions, plus a
calibration check on the predicted-variance head (are 90%-interval
predictions actually right ~90% of the time). Numpy-only.
"""

import math


def _as_lists(y_true, y_pred):
    yt = list(y_true)
    yp = list(y_pred)
    assert len(yt) == len(yp) and len(yt) > 0
    return yt, yp


def mape(y_true, y_pred, eps=1e-9) -> float:
    """Mean Absolute Percentage Error in percent."""
    yt, yp = _as_lists(y_true, y_pred)
    n = len(yt)
    s = 0.0
    for a, p in zip(yt, yp):
        denom = abs(a) if abs(a) > eps else eps
        s += abs((a - p) / denom)
    return 100.0 * s / n


def rmse(y_true, y_pred) -> float:
    yt, yp = _as_lists(y_true, y_pred)
    n = len(yt)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(yt, yp)) / n)


def rmsle(y_true, y_pred, eps=1e-9) -> float:
    """Root Mean Squared Logarithmic Error (log-scale RMSE)."""
    yt, yp = _as_lists(y_true, y_pred)
    n = len(yt)
    s = 0.0
    for a, p in zip(yt, yp):
        la = math.log(max(a, 0.0) + 1.0 + eps)
        lp = math.log(max(p, 0.0) + 1.0 + eps)
        s += (la - lp) ** 2
    return math.sqrt(s / n)


def r2(y_true, y_pred) -> float:
    yt, yp = _as_lists(y_true, y_pred)
    mean = sum(yt) / len(yt)
    ss_tot = sum((a - mean) ** 2 for a in yt)
    ss_res = sum((a - p) ** 2 for a, p in zip(yt, yp))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def calibration_coverage(y_true, mean, log_var, sigma: float = 1.645) -> float:
    """Fraction of targets inside mean +/- sigma*std (default 90% interval).

    mean/log_var are in the same (log-runtime) space as y_true here.
    """
    yt, m, lv = list(y_true), list(mean), list(log_var)
    assert len(yt) == len(m) == len(lv)
    hits = 0
    for a, mu, lvr in zip(yt, m, lv):
        std = math.sqrt(math.exp(lvr))
        if abs(a - mu) <= sigma * std:
            hits += 1
    return hits / len(yt)


def expected_calibration_error(y_true, mean, log_var, n_bins: int = 10) -> float:
    """ECE over predicted interval widths: bin by std, compare empirical coverage."""
    yt, m, lv = list(y_true), list(mean), list(log_var)
    stds = [math.sqrt(math.exp(v)) for v in lv]
    order = sorted(range(len(yt)), key=lambda i: stds[i])
    chunk = max(1, len(yt) // n_bins)
    ece = 0.0
    nb = 0
    for b in range(n_bins):
        idx = order[b * chunk:(b + 1) * chunk]
        if not idx:
            continue
        # Nominal coverage of 1-sigma interval ~68.3%
        hits = sum(1 for i in idx if abs(yt[i] - m[i]) <= stds[i])
        acc = hits / len(idx)
        ece += abs(acc - 0.6827) * (len(idx) / len(yt))
        nb += 1
    return ece


def core_metrics(y_true, y_pred, mean=None, log_var=None) -> dict:
    """All core metrics in one call."""
    out = {
        'mape': mape(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'rmsle': rmsle(y_true, y_pred),
        'r2': r2(y_true, y_pred),
        'n': len(list(y_true)),
    }
    if mean is not None and log_var is not None:
        out['coverage_90'] = calibration_coverage(y_true, mean, log_var, sigma=1.645)
        out['ece_1sigma'] = expected_calibration_error(y_true, mean, log_var)
    return out


if __name__ == '__main__':
    # Deterministic smoke test
    yt = [1.0, 2.0, 4.0, 8.0, 16.0]
    yp = [1.1, 1.9, 4.2, 7.5, 17.0]
    m = core_metrics(yt, yp, mean=[0.0, 0.7, 1.4, 2.1, 2.8],
                     log_var=[-2.0] * 5)
    assert m['mape'] > 0 and m['rmsle'] > 0
    assert 0.0 <= m['coverage_90'] <= 1.0
    print(f"Metrics OK: MAPE={m['mape']:.2f}% RMSLE={m['rmsle']:.4f} R2={m['r2']:.4f} cov90={m['coverage_90']:.2f}")
