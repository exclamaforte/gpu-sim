#!/usr/bin/env python3
"""Repeated-trial statistics and noise-aware target construction.

Turns N raw per-trial timings into a training target:
- Median + IQR (robust)
- Fitted log-normal mean/variance (for Gaussian NLL loss)
- Documented rule for discarding high-variance examples
"""

import math
import statistics
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class DiscardReason(Enum):
    """Reasons for discarding a measurement example."""
    TOO_FEW_TRIALS = "too_few_trials"
    HIGH_VARIANCE = "high_variance"
    OUTLIER_CONTAMINATION = "outlier_contamination"
    BIMODAL_DISTRIBUTION = "bimodal_distribution"
    NONE = "none"


@dataclass
class TrialStats:
    """Statistics computed from repeated trials."""
    n_trials: int
    raw_times_ms: list[float]
    
    # Robust statistics
    median_ms: float
    q1_ms: float
    q3_ms: float
    iqr_ms: float
    mad_ms: float  # Median absolute deviation
    
    # Parametric statistics
    mean_ms: float
    std_ms: float
    cv: float  # Coefficient of variation
    
    # Log-normal fit (for Gaussian NLL)
    log_mean: float
    log_std: float
    
    # Outlier analysis
    outliers_removed: int
    outlier_threshold_ms: float
    
    # Quality flags
    discard_reason: DiscardReason
    is_reliable: bool
    
    # Confidence intervals
    median_ci_95: tuple[float, float]
    mean_ci_95: tuple[float, float]


def compute_trial_stats(
    times_ms: list[float],
    min_trials: int = 10,
    max_cv: float = 0.15,  # 15% coefficient of variation
    outlier_sigma: float = 2.0,
    max_outlier_fraction: float = 0.2,
) -> TrialStats:
    """Compute robust statistics from repeated timing trials.
    
    Args:
        times_ms: List of execution times in milliseconds
        min_trials: Minimum trials required for reliable statistics
        max_cv: Maximum coefficient of variation (std/mean) to trust
        outlier_sigma: Sigma threshold for outlier removal (on log scale)
        max_outlier_fraction: Max fraction of trials that can be outliers
    
    Returns:
        TrialStats with all computed metrics and quality assessment
    """
    n = len(times_ms)
    
    # Check minimum trials
    if n < min_trials:
        return TrialStats(
            n_trials=n,
            raw_times_ms=times_ms,
            median_ms=0, q1_ms=0, q3_ms=0, iqr_ms=0, mad_ms=0,
            mean_ms=0, std_ms=0, cv=0,
            log_mean=0, log_std=0,
            outliers_removed=0, outlier_threshold_ms=0,
            discard_reason=DiscardReason.TOO_FEW_TRIALS,
            is_reliable=False,
            median_ci_95=(0, 0),
            mean_ci_95=(0, 0),
        )
    
    # Sort for percentile calculations
    sorted_times = sorted(times_ms)
    
    # Robust statistics
    median_ms = statistics.median(sorted_times)
    q1_ms = statistics.quantiles(sorted_times, n=4)[0]
    q3_ms = statistics.quantiles(sorted_times, n=4)[2]
    iqr_ms = q3_ms - q1_ms
    mad_ms = statistics.median([abs(x - median_ms) for x in sorted_times])
    
    # Parametric statistics
    mean_ms = statistics.mean(sorted_times)
    std_ms = statistics.stdev(sorted_times) if n > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else float('inf')
    
    # Outlier removal using IQR method (Tukey's fences)
    lower_fence = q1_ms - 1.5 * iqr_ms
    upper_fence = q3_ms + 1.5 * iqr_ms
    filtered_times = [t for t in sorted_times if lower_fence <= t <= upper_fence]
    outliers_removed = n - len(filtered_times)
    outlier_threshold_ms = upper_fence
    
    # If too many outliers, flag as unreliable
    if outliers_removed / n > max_outlier_fraction:
        return TrialStats(
            n_trials=n,
            raw_times_ms=times_ms,
            median_ms=median_ms, q1_ms=q1_ms, q3_ms=q3_ms, iqr_ms=iqr_ms, mad_ms=mad_ms,
            mean_ms=mean_ms, std_ms=std_ms, cv=cv,
            log_mean=0, log_std=0,
            outliers_removed=outliers_removed, outlier_threshold_ms=outlier_threshold_ms,
            discard_reason=DiscardReason.OUTLIER_CONTAMINATION,
            is_reliable=False,
            median_ci_95=(0, 0),
            mean_ci_95=(0, 0),
        )
    
    # Use filtered times for final statistics
    final_times = filtered_times if outliers_removed > 0 else sorted_times
    final_n = len(final_times)
    
    if final_n < 2:
        final_times = sorted_times
        final_n = n
    
    median_ms = statistics.median(final_times)
    mean_ms = statistics.mean(final_times)
    std_ms = statistics.stdev(final_times) if final_n > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else float('inf')
    
    # Log-normal fit (for Gaussian NLL loss)
    log_times = [math.log(t) for t in final_times if t > 0]
    if len(log_times) >= 2:
        log_mean = statistics.mean(log_times)
        log_std = statistics.stdev(log_times)
    else:
        log_mean = math.log(median_ms) if median_ms > 0 else 0
        log_std = 0.0
    
    # Check for bimodality (Hartigan's dip test approximation)
    # Simple heuristic: check if there's a gap in the histogram
    discard_reason = DiscardReason.NONE
    is_reliable = True
    
    # High variance check
    if cv > max_cv:
        discard_reason = DiscardReason.HIGH_VARIANCE
        is_reliable = False
    
    # Bimodality check (simple heuristic)
    if _check_bimodal(final_times):
        discard_reason = DiscardReason.BIMODAL_DISTRIBUTION
        is_reliable = False
    
    # Confidence intervals (using bootstrap for median, t-distribution for mean)
    median_ci_95 = _bootstrap_ci(final_times, statistics.median, n_bootstrap=1000)
    mean_ci_95 = _mean_ci(final_times)
    
    return TrialStats(
        n_trials=n,
        raw_times_ms=times_ms,
        median_ms=median_ms,
        q1_ms=statistics.quantiles(final_times, n=4)[0] if final_n >= 4 else final_times[0],
        q3_ms=statistics.quantiles(final_times, n=4)[2] if final_n >= 4 else final_times[-1],
        iqr_ms=statistics.quantiles(final_times, n=4)[2] - statistics.quantiles(final_times, n=4)[0] if final_n >= 4 else 0,
        mad_ms=statistics.median([abs(x - median_ms) for x in final_times]),
        mean_ms=mean_ms,
        std_ms=std_ms,
        cv=cv,
        log_mean=log_mean,
        log_std=log_std,
        outliers_removed=outliers_removed,
        outlier_threshold_ms=outlier_threshold_ms,
        discard_reason=discard_reason,
        is_reliable=is_reliable,
        median_ci_95=median_ci_95,
        mean_ci_95=mean_ci_95,
    )


def _check_bimodal(times: list[float], bins: int = 20) -> bool:
    """Simple bimodality check using histogram gap detection."""
    if len(times) < 10:
        return False
    
    hist, bin_edges = np_histogram(times, bins=bins)
    # Look for significant gaps between peaks
    max_count = max(hist)
    if max_count == 0:
        return False
    
    # Normalize
    hist_norm = [h / max_count for h in hist]
    
    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(hist_norm) - 1):
        if hist_norm[i] > hist_norm[i-1] and hist_norm[i] > hist_norm[i+1]:
            if hist_norm[i] > 0.1:  # At least 10% of max
                peaks.append(i)
    
    # If two well-separated peaks, might be bimodal
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            gap = peaks[i+1] - peaks[i]
            if gap > bins * 0.3:  # Peaks separated by >30% of range
                return True
    
    return False


def _bootstrap_ci(data: list[float], stat_func, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    import random
    if len(data) < 2:
        val = stat_func(data)
        return (val, val)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [random.choice(data) for _ in data]
        bootstrap_stats.append(stat_func(sample))
    
    bootstrap_stats.sort()
    alpha = (1 - ci) / 2
    lower_idx = int(alpha * n_bootstrap)
    upper_idx = int((1 - alpha) * n_bootstrap)
    
    return (bootstrap_stats[lower_idx], bootstrap_stats[upper_idx])


def _mean_ci(data: list[float], ci: float = 0.95) -> tuple[float, float]:
    """t-distribution confidence interval for mean."""
    if len(data) < 2:
        val = statistics.mean(data) if data else 0
        return (val, val)
    
    mean = statistics.mean(data)
    se = statistics.stdev(data) / math.sqrt(len(data))
    
    # t-critical value approximation (for 95% CI, t ≈ 2 for n>30)
    try:
        from scipy import stats as scipy_stats
        t_crit = scipy_stats.t.ppf((1 + ci) / 2, len(data) - 1)
    except ImportError:
        t_crit = 1.96 if len(data) > 30 else 2.5  # Approximation
    
    margin = t_crit * se
    return (mean - margin, mean + margin)


def np_histogram(data: list[float], bins: int = 10):
    """Simple numpy-free histogram."""
    if not data:
        return [0] * bins, [0.0] * (bins + 1)
    
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val:
        return [len(data)] + [0] * (bins - 1), [min_val] * (bins + 1)
    
    bin_width = (max_val - min_val) / bins
    edges = [min_val + i * bin_width for i in range(bins + 1)]
    hist = [0] * bins
    
    for val in data:
        if val == max_val:
            idx = bins - 1
        else:
            idx = int((val - min_val) / bin_width)
        hist[idx] += 1
    
    return hist, edges


def construct_training_target(stats: TrialStats, target_type: str = "log_normal") -> dict:
    """Construct training target from trial statistics.
    
    Args:
        stats: TrialStats from compute_trial_stats
        target_type: "median", "mean", "log_normal", or "both"
    
    Returns:
        Dictionary with target values for model training
    """
    target = {
        'reliable': stats.is_reliable,
        'discard_reason': stats.discard_reason.value,
        'n_trials': stats.n_trials,
        'outliers_removed': stats.outliers_removed,
        'cv': stats.cv,
    }
    
    if target_type in ("median", "both"):
        target.update({
            'target_median_ms': stats.median_ms,
            'target_iqr_ms': stats.iqr_ms,
            'target_median_ci_lower': stats.median_ci_95[0],
            'target_median_ci_upper': stats.median_ci_95[1],
        })
    
    if target_type in ("mean", "both"):
        target.update({
            'target_mean_ms': stats.mean_ms,
            'target_std_ms': stats.std_ms,
            'target_mean_ci_lower': stats.mean_ci_95[0],
            'target_mean_ci_upper': stats.mean_ci_95[1],
        })
    
    if target_type in ("log_normal", "both"):
        target.update({
            'target_log_mean': stats.log_mean,
            'target_log_std': stats.log_std,
        })
    
    return target


def process_measurement_batch(
    measurements: list[dict],
    min_trials: int = 30,
    max_cv: float = 0.15,
) -> list[dict]:
    """Process a batch of measurements, each with repeated trials.
    
    Args:
        measurements: List of dicts with keys: kernel_hash, launch_config, trials_ms
        min_trials: Minimum trials per kernel
        max_cv: Maximum CV for reliable measurement
    
    Returns:
        List of processed measurements with training targets
    """
    results = []
    
    for m in measurements:
        times = m.get('trials_ms', [])
        stats = compute_trial_stats(times, min_trials=min_trials, max_cv=max_cv)
        target = construct_training_target(stats, target_type="log_normal")
        
        result = {
            'kernel_hash': m['kernel_hash'],
            'launch_config': m['launch_config'],
            **target,
            'raw_trials': times,
        }
        results.append(result)
    
    return results


if __name__ == '__main__':
    # Demo with synthetic data
    import random
    
    # Simulate measurements with some noise
    base_time = 5.0  # ms
    for noise_level in [0.02, 0.05, 0.10, 0.20, 0.50]:
        times = [base_time * random.lognormvariate(0, noise_level) for _ in range(30)]
        stats = compute_trial_stats(times)
        target = construct_training_target(stats)
        
        print(f"Noise σ={noise_level:.2f}: median={stats.median_ms:.3f}ms, "
              f"CV={stats.cv:.3f}, log_std={stats.log_std:.3f}, "
              f"reliable={stats.is_reliable}, outliers={stats.outliers_removed}, "
              f"discard={stats.discard_reason.value}")