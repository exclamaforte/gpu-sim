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


def calibration_coverage(y_true, mean, var, sigma: float = 1.645,
                         meas_sigma=0.0) -> float:
    """Fraction of targets inside mean +/- sigma*std (default 90% interval).

    gpu-sim-u2o.10.1 CORRECTED: `var` is a VARIANCE (positive) -- exactly
    what DualHeadPredictor emits (see 9.3 contract). Was `log_var` with an
    exp() inside: feeding the head's output exp()'d a variance (wrong
    intervals). No in-repo callers existed, so the parameter was renamed,
    not shimmed. meas_sigma adds the 6.6 per-cell floor in quadrature.
    """
    yt, m, v = list(y_true), list(mean), list(var)
    assert len(yt) == len(m) == len(v)
    ms = list(meas_sigma) if isinstance(meas_sigma, (list, tuple)) \
        else [meas_sigma] * len(yt)
    hits = 0
    for a, mu, vv, s in zip(yt, m, v, ms):
        std = math.sqrt(vv + s * s)
        if abs(a - mu) <= sigma * std:
            hits += 1
    return hits / len(yt)


def expected_calibration_error(y_true, mean, var, n_bins: int = 10,
                               meas_sigma=0.0) -> float:
    """ECE over predicted interval widths: bin by std, compare empirical
    coverage. Same 10.1 variance-contract correction as coverage above."""
    yt, m, v = list(y_true), list(mean), list(var)
    ms = list(meas_sigma) if isinstance(meas_sigma, (list, tuple)) \
        else [meas_sigma] * len(yt)
    stds = [math.sqrt(vv + s * s) for vv, s in zip(v, ms)]
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


def core_metrics(y_true, y_pred, mean=None, var=None,
                 meas_sigma=0.0) -> dict:
    """All core metrics in one call (10.1: variance contract, see above)."""
    out = {
        'mape': mape(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'rmsle': rmsle(y_true, y_pred),
        'r2': r2(y_true, y_pred),
        'n': len(list(y_true)),
    }
    if mean is not None and var is not None:
        out['coverage_90'] = calibration_coverage(y_true, mean, var,
                                                  sigma=1.645,
                                                  meas_sigma=meas_sigma)
        out['ece_1sigma'] = expected_calibration_error(y_true, mean, var,
                                                       meas_sigma=meas_sigma)
    return out


if __name__ == '__main__':
    # Deterministic smoke test
    yt = [1.0, 2.0, 4.0, 8.0, 16.0]
    yp = [1.1, 1.9, 4.2, 7.5, 17.0]
    m = core_metrics(yt, yp, mean=[0.0, 0.7, 1.4, 2.1, 2.8],
                     var=[0.25] * 5)
    assert m['mape'] > 0 and m['rmsle'] > 0
    assert 0.0 <= m['coverage_90'] <= 1.0
    print(f"Metrics OK: MAPE={m['mape']:.2f}% RMSLE={m['rmsle']:.4f} R2={m['r2']:.4f} cov90={m['coverage_90']:.2f}")
