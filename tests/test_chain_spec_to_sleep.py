"""Chain test: a spec must reach the simulator as a slept, size-aware cost.

Four defects found on 2026-10-23 each broke exactly one link of
    spec -> latency_ms_for_spec -> table -> FAKEGPU_KERNEL_LATENCY_MS
         -> parse_double -> fg_sleep_ns -> observed kernel cost
and every one of them survived a fully green suite, because each link was
tested in isolation and the chain was not:

  gpu-sim-vr1  C parsed the env var as a long -> every sub-ms value slept 0
  gpu-sim-feo  out-of-grid predictions were clamped with no signal
  gpu-sim-el4  the default table had no size axis, so size was ignored
  gpu-sim-371  a "quarantine" disabled 14 passing tests -> suite green

This test walks the real path in a fresh process per measurement -- it builds
the environment exactly as epic_u2o/safe_harness.py does, rather than setting
the env var to a hand-picked literal. That distinction is the point: setting the
env var directly (as tests/test_fakegpu_kernel_latency_subms.py does, correctly,
for its own purpose) proves the C side sleeps what it is given; it cannot notice
a broken table, a size-blind default, or a mangled value upstream.

WHY 2**28 AND NOT 2**24: the calibrated difference between relu[2**10] and
relu[2**24] is 0.072 ms, which is the same order as the ~0.06 ms cost of the
ctypes call and its jitter -- measured across 8 samples that difference fell to
0.036 ms, i.e. below any sane threshold, so a test written at 2**24 flakes.
At 2**28 the calibrated value is 1.408 ms and the measured difference was
1.407-1.452 ms across 8 samples, a 2.8x margin over the 0.5 ms threshold.
Choosing the measurement point so the property dominates the noise is the same
reasoning that made the queue_state_cost assertion differential.

Each measurement runs in its OWN PROCESS: dlopen caches by path, so a second
CDLL of the same .so in one process keeps the config parsed from the first env
value, and a shared-process version silently measures one value twice.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.op_latency import latency_ms_for  # noqa: E402

LIB = REPO / "libfakegpu.so"

SMALL = 2 ** 10
LARGE = 2 ** 28          # largest calibrated size; large enough to dominate noise
MIN_SEPARATION_MS = 0.5  # measured margin is ~1.41 ms, so this is ~2.8x slack

_PROBE = r'''
import ctypes, os, sys, time
from pathlib import Path

repo = Path(sys.argv[2])
sys.path.insert(0, str(repo))
from epic_u2o.op_latency import latency_ms_for_spec

spec = {"op_type": "relu", "shapes": [int(sys.argv[1])], "dtype_bytes": 4}
os.environ["FAKEGPU_KERNEL_LATENCY_MS"] = str(latency_ms_for_spec(spec))

lib = ctypes.CDLL(str(repo / "libfakegpu.so"))
lib.fg_simulate_kernel()          # warm: config read must not be timed
t0 = time.perf_counter()
lib.fg_simulate_kernel()
print((time.perf_counter() - t0) * 1000.0)
'''


def _observed_ms(size: int) -> float:
    assert LIB.exists(), "libfakegpu.so must be built (run make)"
    env = {k: v for k, v in os.environ.items() if k != "GPUSIM_OP_LATENCY_JSON"}
    run = subprocess.run(
        [sys.executable, "-c", _PROBE, str(size), str(REPO)],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert run.returncode == 0, run.stderr[-400:]
    return float(run.stdout.strip())


def test_size_reaches_the_sleep_through_the_whole_chain():
    small = _observed_ms(SMALL)
    large = _observed_ms(LARGE)

    # (a) THE LOAD-BEARING ASSERTION. If any link drops the size -- truncation,
    # a size-blind default, a stale table -- both specs collapse to the same
    # cost and this fails. "Both non-zero" would not catch that.
    assert large > small + MIN_SEPARATION_MS, (
        f"relu[{LARGE}] slept {large:.3f} ms and relu[{SMALL}] slept "
        f"{small:.3f} ms; the calibrated values differ by "
        f"{latency_ms_for('relu', LARGE) - latency_ms_for('relu', SMALL):.3f} "
        f"ms, so the size is being lost somewhere in "
        f"spec -> table -> env -> parse -> sleep"
    )

    # (b) The large case lands near its calibrated value. Loose enough for the
    # ~0.06 ms ctypes overhead, tight enough that a 2x scale error fails.
    cal_large = latency_ms_for("relu", LARGE)
    assert abs(large - cal_large) <= 0.3, (
        f"relu[{LARGE}] is calibrated at {cal_large:.3f} ms but the simulator "
        f"spent {large:.3f} ms"
    )
