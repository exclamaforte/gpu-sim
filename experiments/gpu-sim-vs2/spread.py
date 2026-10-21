"""Spread statistic reused from experiments/gpu-sim-m3d.10/verdict.py.

Steps (identical to verdict.py's proxy derivation):
  1. drop the first value (init iteration overhead)
  2. drop any value > PROXY_TIMER_ARTIFACT_US (1e9 us; a timer artifact)
  3. spread = max(remaining) - min(remaining)

Constants copied verbatim from verdict.py so both code paths agree.
"""
PROXY_TIMER_ARTIFACT_US = 1.0e9


def spread_us(values):
    """values: list of per-iteration durations in microseconds."""
    if len(values) < 3:
        raise ValueError("need at least 3 values")
    post_init = list(values)[1:]
    kept = [v for v in post_init if v <= PROXY_TIMER_ARTIFACT_US]
    if len(kept) < 2:
        raise ValueError("too few values after dropping init/artifacts")
    return max(kept) - min(kept)


def kept_count(values):
    post_init = list(values)[1:]
    return len([v for v in post_init if v <= PROXY_TIMER_ARTIFACT_US])
