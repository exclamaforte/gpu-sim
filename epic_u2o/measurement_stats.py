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
    CONTENTION_BIAS = "contention_bias"  # gpu-sim-u2o.6.6: co-tenancy is a
    # systematic multiplier, not trial noise; it must discard, never average in.
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


# ---------------------------------------------------------------------------
# gpu-sim-u2o.6.6.7 -- CROSS-RUN SPREAD, and what a comparison may conclude.
#
# Everything above this line describes WITHIN one run: N trials of one
# configuration. This section describes the other thing, which is how far the
# SAME configuration moves BETWEEN runs. Both were needed and only the first
# existed.
#
# Two instances forced it:
#   * a +3.9% single-replicate reading that a three-replicate repeat showed was
#     inside the noise (gpu-sim-u2o.6.6.1 rework 3);
#   * the v6 anchor re-run as v8: 106 comparable cells, median 1.018, p90 1.235,
#     max 1.559 (gpu-sim-u2o.6.6.2 variance_probe.json).
#
# So a single-run ratio against a baseline resolves NOTHING smaller than the
# baseline's own spread, and nothing in the workflow forced a comparison to say
# so. This makes it say so.
# ---------------------------------------------------------------------------


@dataclass
class SpreadRecord:
    """A baseline's measured run-to-run spread, versioned WITH the baseline.

    Versioned here for the same reason anchor_version exists: a spread measured
    on one box, one queue state and one driver is a property of THAT baseline,
    not a constant to be reused. Carry spread_version beside the baseline's own
    version and quote both.
    """
    spread_version: int
    baseline_id: str
    n_runs: int
    n_cells: int
    run_job_ids: list
    p10: float
    median: float
    p90: float
    minimum: float
    maximum: float
    statement: str = ""
    # How many ratios each config contributed. With n_runs == 2 this is 1, and a
    # quantile over one ratio per config is a quantile over single samples, not
    # over a distribution -- which is why it is carried rather than implied.
    ratios_per_config: int = 1

    def band(self) -> tuple:
        """The resolution window: ratios inside this are indistinguishable
        from run-to-run movement of the identical code."""
        return (self.p10, self.p90)

    def to_dict(self) -> dict:
        return {
            "spread_version": self.spread_version,
            "baseline_id": self.baseline_id,
            "n_runs": self.n_runs,
            "n_cells": self.n_cells,
            "ratios_per_config": self.ratios_per_config,
            "run_job_ids": list(self.run_job_ids),
            "ratio_quantiles": {"min": self.minimum, "p10": self.p10,
                                "median": self.median, "p90": self.p90,
                                "max": self.maximum},
            "resolution_band": list(self.band()),
            "statement": self.statement,
        }


def _quantile(sorted_xs, p: float) -> float:
    if not sorted_xs:
        raise ValueError("no values to take a quantile of")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    k = (len(sorted_xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def spread_from_runs(runs: list, baseline_id: str, spread_version: int = 1) -> SpreadRecord:
    """Measure a baseline's cross-run spread from >=2 runs of the same configs.

    `runs` is a list of (job_id, {config_label: value}) pairs. Configs are
    joined on the label the measurement itself emits, so the join key is the
    measurement's, not this function's.

    Fewer than two runs is REFUSED rather than defaulted. A spread from one run
    is zero by construction -- there is no run-to-run movement to measure -- and
    returning 0.0 would hand every later comparison a band of zero width and let
    any difference look significant. That is the failure this whole module
    exists to prevent.
    """
    if len(runs) < 2:
        raise ValueError(
            "cross-run spread needs at least 2 runs; a single run has no "
            "run-to-run movement to measure, and returning 0.0 would make "
            "every later comparison look significant")
    labels = None
    per_label = {}
    job_ids = []
    for job_id, cells in runs:
        job_ids.append(job_id)
        if labels is None:
            labels = set(cells)
        else:
            labels &= set(cells)
        for k, v in cells.items():
            per_label.setdefault(k, []).append(v)
    ratios = []
    for k in sorted(labels):
        vals = per_label[k]
        # SIGNED, BASELINE-RELATIVE ratios. Not max/min.
        #
        # max/min is symmetric and one-sided: it is always >= 1, so a config
        # that got FASTER is relabelled as a slow envelope. On the real anchor
        # the minimum cell moved 14.8% FASTER (v8/v6 = 0.8523) and max/min would
        # have reported that as 1.1745 -- the wrong direction, on the one
        # artifact whose purpose is to stop exactly that.
        #
        # Relative to the FIRST run (the baseline), every later run contributes
        # v_i / v_0. That is the same statistic the variance probe uses, so
        # p10 can fall below 1, `min` means what its name says, and effect_faster
        # is judged against real fast-side noise instead of being unreachable.
        base = vals[0]
        if base:
            ratios.extend(v / base for v in vals[1:])

    if not ratios:
        raise ValueError("no comparable non-zero configs across the runs given")
    xs = sorted(ratios)
    return SpreadRecord(
        spread_version=spread_version,
        baseline_id=baseline_id,
        n_runs=len(runs),
        n_cells=len(ratios),
        ratios_per_config=len(runs) - 1,
        run_job_ids=job_ids,
        p10=round(_quantile(xs, 0.10), 4),
        median=round(_quantile(xs, 0.50), 4),
        p90=round(_quantile(xs, 0.90), 4),
        minimum=round(xs[0], 4),
        maximum=round(xs[-1], 4),
        statement=(f"measured over {len(runs)} runs and {len(ratios)} configs; "
                   f"a single-run ratio against {baseline_id} cannot resolve "
                   f"anything inside [{round(_quantile(xs, 0.10), 4)}, "
                   f"{round(_quantile(xs, 0.90), 4)}]"),
    )


def verdict_for_ratio(ratio: float, spread: Optional[SpreadRecord]) -> dict:
    """What may be concluded about `ratio` against a baseline.

    REPORTS. Never raises, and never returns a bare "effect" it cannot support.

      effect_slower      ratio above the baseline's p90
      effect_faster      ratio below its p10
      inconclusive       ratio inside [p10, p90] -- indistinguishable from the
                         same code moving between runs
      no_spread_recorded there is no spread for this baseline, so NO resolution
                         is claimed. This is deliberately NOT inconclusive and
                         NOT an effect: it is the absence of the information
                         needed to tell, and it must not read as a pass.

    The inconclusive case is the point of the module. It is also why this must
    not become a gate that fails everything -- the v6 anchor's own p90 is 1.235,
    so a hard rule like "any ratio under 1.2x is inconclusive" would retire most
    of this lab's legitimate A/B results. The band is per-baseline and measured.
    """
    if spread is None:
        return {
            "ratio": ratio,
            "verdict": "no_spread_recorded",
            "resolution_band": None,
            "explanation": ("no measured cross-run spread for this baseline, so "
                            "this ratio cannot be called an effect or noise. "
                            "Record one (>=2 runs) before concluding anything."),
        }
    lo, hi = spread.band()
    if ratio > hi:
        verdict = "effect_slower"
    elif ratio < lo:
        verdict = "effect_faster"
    else:
        verdict = "inconclusive"
    return {
        "ratio": ratio,
        "verdict": verdict,
        "resolution_band": [lo, hi],
        "spread_version": spread.spread_version,
        "n_runs": spread.n_runs,
        "n_cells": spread.n_cells,
        "explanation": (
            f"ratio {ratio} vs baseline {spread.baseline_id} "
            f"(spread v{spread.spread_version}, {spread.n_runs} runs, "
            f"{spread.n_cells} configs, band [{lo}, {hi}]): " +
            ("indistinguishable from run-to-run movement of the same code"
             if verdict == "inconclusive" else
             f"outside the band, so the shift exceeds measured run-to-run noise")),
    }


# ---------------------------------------------------------------------------
# gpu-sim-u2o.6.6 -- noise-aware target construction on REAL data.
#
# The audit (gpu-sim-u2o.15) found this module's __main__ demo ran on synthetic
# random.lognormvariate draws, not real trial statistics. This section is the
# real-data path, and it encodes the PI rulings that govern it:
#
#   * regime: the informative split is 10 us (PI 2026-09-27). Trial-noise sigma
#     is regime-dependent (~7% sub-10us, ~0.1% over-10us) -- a single sigma
#     learns the mixture and is wrong for both, so 9.3 must calibrate per regime.
#   * cell median = median of PER-INVOCATION medians, quoted with its invocation
#     count and cell (acceptance criterion 2). A single run can sit far from its
#     cell's median (1.472 / 2.208 / 2.304 us on 6.6.1).
#   * noise vs bias separate (criterion 4): trial noise averages away, co-tenancy
#     bias does not. Contention discards via ContentionBias; it never widens sigma.
#   * every number carries its population and convention (criterion 3).
# ---------------------------------------------------------------------------

REGIME_CUTOFF_MS = 0.01  # 10 us, PI ruling 2026-09-27

# gpu-sim-u2o.6.6.3 discard thresholds, quoted verbatim in the units measured.
# Derived idle-vs-saturated-burner; insensitive, not delicate.
CONTENTION_POWER_W = 150.0
CONTENTION_UTIL_PCT = 50.0
CONTENTION_TEMP_C = 50.0


def regime_of(median_ms: float) -> str:
    """Informative-regime split (PI 2026-09-27): below 10 us the labels are
    launch-dominated (~7% across-invocation noise); at/above 10 us they are
    reproducible to ~0.1%."""
    return "sub-10us" if median_ms < REGIME_CUTOFF_MS else "over-10us"


def _num(text) -> Optional[float]:
    """Leading float of a telemetry field ('574.3 W' -> 574.3); None if absent."""
    if text is None:
        return None
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError):
        return None


def apply_contention_rule(telemetry_before: dict, telemetry_after: dict,
                          queue_before: dict, queue_after: dict) -> dict:
    """Apply the gpu-sim-u2o.6.6.3 discard rule to one family's boundary records.

    Returns a disposition dict (JSON-safe). Fields the telemetry schema does not
    carry are listed under coverage_gaps -- a stated omission, never a null
    pretending to be coverage. Channels per the 6.6.3 rule: queue_state is
    primary for queue-routed contention, telemetry for the rest; neither alone
    is sufficient, and queue_state is blind to unqueued tenants by measurement.
    """
    fired = []
    gaps = []
    readings = {}
    # PI ruling 2026-10-06 (gpu-sim-u2o.6.6): temperature reads the BEFORE
    # (ambient) boundary only. Post-run heat is our own workload's exhaust --
    # the anchor's gemm family read 44C before / 53C after 9 s of its own
    # hottest kernels with a clean queue, and a clause that fires on that
    # discards the reference for being measured. Power/util stay
    # either-boundary: a co-tenant burns power throughout the run.
    observed = {}
    for sig, thresh, unit, boundaries in (
            ("power.draw", CONTENTION_POWER_W, "W", ("before", "after")),
            ("utilization.gpu", CONTENTION_UTIL_PCT, "%", ("before", "after")),
            ("temperature.gpu", CONTENTION_TEMP_C, "C", ("before",))):
        src = {"before": telemetry_before, "after": telemetry_after}
        vals = {b: _num(src[b].get(sig)) for b in boundaries}
        readings[sig] = {b: vals.get(b) for b in ("before", "after")}
        if all(v is None for v in vals.values()):
            gaps.append(f"{sig}: absent from telemetry schema, clause unevaluated")
            continue
        present = [v for v in vals.values() if v is not None]
        worst = max(present)
        if worst > thresh:
            where = sorted(b for b, v in vals.items() if v == worst)
            scope = "either-boundary" if len(boundaries) > 1 else "before-only"
            fired.append(f"{sig} {worst} {unit} exceeds {thresh} {unit} "
                         f"at {'+'.join(where)} ({scope} clause)")
    q = {}
    for name, rec in (("before", queue_before), ("after", queue_after)):
        q[name] = {
            "n_other_active": rec.get("n_other_active"),
            "n_other_running_at_sample_time": rec.get(
                "n_other_running_at_sample_time"),
            "own_job_id_present": rec.get("own_job_id") is not None,
            "own_job_id": rec.get("own_job_id"),
            "own_job_entry_present": rec.get("own_job_entry") is not None,
        }
    running = [q[b]["n_other_running_at_sample_time"] for b in ("before", "after")]
    if any(r is not None and r > 0 for r in running):
        fired.append(f"queue_state n_other_running_at_sample_time={running}: "
                     "a queue-routed tenant was running at sample time")
    discard = bool(fired)
    return {
        "disposition": "discard" if discard else "keep",
        "discard_reason": (DiscardReason.CONTENTION_BIAS.value
                             if discard else DiscardReason.NONE.value),
        "fired_clauses": fired,
        "signal_readings": readings,
        "coverage_gaps": gaps,
        "queue_channel": q,
        "population": ("one family's boundary records: telemetry_before/after "
                         "plus queue_state_before/after"),
        "convention": ("discard on ANY fired clause (bias, systematic); gaps "
                         "are stated omissions and never count as passes; "
                         "temperature reads the BEFORE (ambient) boundary only "
                         "per PI ruling 2026-10-06 -- post-run heat is our own "
                         "workload's exhaust, not a co-tenant"),
    }


def construct_cell_target(label: str, medians_across_invocations_ms: list[float],
                          n_trials: int, regime_sigma: dict,
                          contention: dict) -> dict:
    """Noise-aware training target for one anchor cell.

    target_median_ms is the median of the PER-INVOCATION medians (criterion 2),
    never a single run. trial_noise_sigma is the REGIME sigma (reducible noise);
    co-tenancy is a separate bias channel that discards via contention, and the
    two are never mixed into one sigma (criterion 4). Every value states its
    population and convention (criterion 3).
    """
    meds = sorted(medians_across_invocations_ms)
    target = statistics.median(meds)
    regime = regime_of(target)
    sigma = regime_sigma[regime]
    keep = contention.get("disposition", "keep") == "keep"
    return {
        "cell": label,
        "n_trials_per_invocation": n_trials,
        "n_invocations": len(meds),
        "target_median_ms": target,
        "invocation_spread": {"min_ms": meds[0], "max_ms": meds[-1],
                                "max_min_ratio": (meds[-1] / meds[0]
                                                   if meds[0] else None)},
        "regime": regime,
        "trial_noise_sigma": sigma["cv"],
        "trial_noise_sigma_population": sigma["population"],
        "observed_across_invocation_cv": None,  # joined by the driver from
        # the anchor record; None here means not yet joined, not zero.
        "bias_disposition": contention.get("disposition"),
        "bias_fired_clauses": contention.get("fired_clauses", []),
        "bias_coverage_gaps": contention.get("coverage_gaps", []),
        "discard": not keep,
        "discard_reason": contention.get(
            "discard_reason", DiscardReason.NONE.value),
        "population": (f"{len(meds)} per-invocation medians of {n_trials} "
                         f"cudaEvent trials each, cell '{label}'"),
        "convention": ("target = median of per-invocation medians, ms; noise "
                         "= regime CV (reducible); bias = contention channel "
                         "(discards, never widens noise)"),
    }


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