"""A sub-millisecond calibrated kernel latency must actually be slept.

gpu-sim-vr1: FAKEGPU_KERNEL_LATENCY_MS was parsed with strtol, so every value
below 1.0 became 0 and the calibrated table -- most of which is sub-millisecond
-- was discarded at the process boundary while the library reported a size-aware
number.

This has to be a TIMING assertion, not a flag or struct check. The original test
asserted fg_latency_active(), which is set whenever the sleep branch is entered;
it would still pass if the sleep used the wrong unit (e.g. ms*1000 reads as
0.25 us instead of 250 us), because the branch is taken either way.

The value is 0.25 ms rather than the table's smallest entry on purpose: a real
0.000893 ms sleep is ~0.9 us, far below the ~70 us cost of the ctypes call, so
wall clock cannot distinguish it from zero. 0.25 ms is unambiguously below the
1.0 ms truncation boundary AND far above measurement noise, which makes the
regression decisive: the truncating code sleeps nothing (~11 us), the fixed code
sleeps 250 us.

Each measurement runs in its OWN PROCESS. dlopen caches by path, so a second
CDLL of the same .so in one process keeps the config parsed from the first env
value -- a shared-process version of this test measures the first value twice.

Measured on master after the fix (fresh process per value, this machine):
    0.25   -> 316.2 us   (250 + ~66 us overhead)
    0.75   -> 818.1 us   (750 + ~68 us overhead)
    0      ->  10.9 us   (no sleep)
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "libfakegpu.so"

_PROBE = """
import ctypes, json, time
lib = ctypes.CDLL({lib!r})
lib.fg_simulate_kernel()          # warm: config read must not be timed
t0 = time.perf_counter_ns()
lib.fg_simulate_kernel()
print(json.dumps({{"elapsed_ns": time.perf_counter_ns() - t0}}))
"""


def _slept_ns(latency_ms: str) -> int:
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(lib=str(LIB))],
        capture_output=True, text=True, timeout=60,
        env={"FAKEGPU_KERNEL_LATENCY_MS": latency_ms, "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr[:400]
    return json.loads(out.stdout.strip())["elapsed_ns"]


def test_submillisecond_kernel_latency_is_actually_slept():
    elapsed = _slept_ns("0.25")
    assert elapsed >= 150_000, (
        f"requested 0.25 ms (250000 ns) of simulated kernel latency but the "
        f"call returned in {elapsed / 1000:.1f} us -- the fractional part is "
        f"being truncated to an integer millisecond again, which zeroes every "
        f"sub-millisecond entry in the calibration table"
    )


def test_no_latency_requested_sleeps_nothing():
    """The floor above must be distinguishable from 'no sleep at all'."""
    elapsed = _slept_ns("0")
    assert elapsed < 150_000, (
        f"no latency was requested but the call took {elapsed / 1000:.1f} us; "
        f"the sub-millisecond assertion above would then be vacuous"
    )


def test_whole_millisecond_latency_keeps_its_fraction():
    """The fix must not disturb the integer case that used to work."""
    elapsed = _slept_ns("11.282289")
    assert elapsed >= 11_150_000, (
        f"requested 11.282289 ms but the call returned in {elapsed / 1000:.0f} "
        f"us -- the fractional part is being dropped again (11 ms truncated)"
    )
