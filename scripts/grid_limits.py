"""Shared grid-capping rule for bk9 gate (gpu-sim-4r9k).

Single source of truth: both measure.py (8zqr) and parse_powered.py import this.
The boundary test (tests/test_grid_capped_boundary.py) also imports it, so a
future copy-paste divergence fails the suite.
"""
PER_DIM_LIMITS = (2**31 - 1, 65535, 65535)


def grid_capped(grid):
    """True iff any grid dimension exceeds its per-dimension limit."""
    g = list(grid) + [1] * (3 - len(grid))
    return (g[0] > PER_DIM_LIMITS[0]) or (g[1] > PER_DIM_LIMITS[1]) or (g[2] > PER_DIM_LIMITS[2])
