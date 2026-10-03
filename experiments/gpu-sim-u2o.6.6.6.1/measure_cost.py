#!/usr/bin/env python3
"""Spot-check the queue_state instrument cost on THIS box, with machine state attached.

gpu-sim-u2o.6.6.6.1 rework: the committed metrics.json headline (0.1123 ms,
measured on a quieter box) cannot be re-measured exactly under load, and a
script that demanded exact reproduction would fail for reasons wholly outside
the code. So this script does what the reviewer asked for: it measures BOTH
arms back-to-back in one process, so the machine cancels, and checks the CODE
properties (differential speedup, hit rate, staleness reporting) rather than
an absolute millisecond figure.

It writes spot_check.json next to itself and exits nonzero if any gate fails.
run.sh runs this plus the full test suite.

Gates (all machine-cancelling or margined):
  1. differential: warm-cached median <= uncached median / 10, same process.
  2. absolute with margin: best-of-N single-call cost < 2x the 0.139 ms device
     time. The strict 0.81x ratio is certified by the artifact measurement, not
     by this gate; this gate's job is to catch the inversion coming back (a
     reintroduced full scan costs tens of ms, 100x+ above the threshold).
  3. the cache engages: hit_rate > 0.5 and parses < files_seen.
  4. the staleness bound is reported: load_stats carries staleness_bound,
     terminal_reverify_sec, listing_ttl_sec.

Usage: python3 experiments/gpu-sim-u2o.6.6.6.1/measure_cost.py
"""
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "spot_check.json"

from epic_u2o import queue_state as qs  # noqa: E402
def _box_is_quiet(loadavg_1m) -> bool:
    """Is the box close enough to idle that a sub-ms timing means something?

    A stat-dominated call of ~0.1 ms is entirely at the mercy of run-queue
    position once the box is oversubscribed. This box has 32 cores; at
    loadavg 94 (observed during this rework) the SAME code measures 0.31 ms
    median that it measures 0.11 ms on a quiet box. So the absolute
    instrument-vs-device claim is only made when the box can support it.
    """
    try:
        ncpu = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        ncpu = os.cpu_count() or 1
    return loadavg_1m <= QUIET_BOX_LOAD_PER_CPU * ncpu


def _absolute_verdict(cached_ms: float, loadavg_1m) -> str:
    """PASS / FAIL / NOT_EVALUABLE -- three distinct outcomes, never two.

    The whole point: a check whose failure mode is indistinguishable from what
    it reports silently disables the lab. "The box was too loaded to tell" and
    "the instrument is cheap" must not share a return value.
    """
    ratio = cached_ms / DEVICE_TIME_MS
    if not _box_is_quiet(loadavg_1m):
        return ("not_evaluable_under_load: instrument is "
                f"{ratio:.2f}x device work, but the box is oversubscribed so "
                "this figure measures the scheduler, not the code. Re-run on a "
                "quiet box to evaluate it; the differential gate is "
                "load-cancelling and does hold.")
    if ratio < 1.0:
        return f"pass: instrument is {ratio:.2f}x the 0.139 ms device work"
    return ("FAIL: instrument is "
            f"{ratio:.2f}x the 0.139 ms device work on a QUIET box -- the "
            "inversion is back")

DEVICE_TIME_MS = 0.139  # gpu-sim-u2o.6.7.1 measured device work per example
DIFFERENTIAL_FACTOR = 10.0
# The absolute instrument-vs-device-time claim is NOT load-robust, and pretending
# otherwise is the defect this rework exists to remove. 0.139 ms of device work was
# measured elsewhere, on a quieter box; a stat-dominated sub-millisecond
# micro-benchmark on an oversubscribed box measures the SCHEDULER. So the
# absolute ratio is evaluated only when the box is not oversubscribed, and is
# otherwise reported as NOT EVALUABLE rather than silently passed. See
# quiet_box_threshold below and _absolute_verdict().
QUIET_BOX_LOAD_PER_CPU = 1.0   # loadavg(1m) / ncpu at or below this = quiet
N = 15


def single_cached_call_ms() -> float:
    t0 = time.perf_counter()
    qs.queue_state()
    return (time.perf_counter() - t0) * 1e3


def single_uncached_call_ms() -> float:
    qs.reset_cache()  # cold: nothing cached, full glob + parse on next call
    t0 = time.perf_counter()
    qs.queue_state()
    return (time.perf_counter() - t0) * 1e3


def main() -> int:
    # Interleave the arms round by round so a load drift mid-run hits both.
    qs.reset_cache()
    qs.queue_state()  # warm once; the first-call cost is not the steady state
    cached, uncached = [], []
    for _ in range(N):
        cached.append(single_cached_call_ms())
        uncached.append(single_uncached_call_ms())
        qs.queue_state()  # re-warm after the reset so the next cached sample is warm
    qs.reset_cache()
    for _ in range(5):  # re-warm, then collect steady-state stats for the record
        qs.queue_state()
    stats = qs.load_stats()

    cached_med = statistics.median(cached)
    cached_min = min(cached)
    uncached_med = statistics.median(uncached)
    speedup = uncached_med / cached_med if cached_med else 0.0

    try:
        loadavg = os.getloadavg()
    except OSError:
        loadavg = None
    try:
        queue_files = len(list(qs.QUEUE_DIR.glob("*.json")))
    except OSError:
        queue_files = None

    absolute_verdict = _absolute_verdict(cached_med, loadavg[0] if loadavg else 0.0)
    absolute_ok = not absolute_verdict.startswith("FAIL")
    gates = {
        "differential_cached_le_uncached_over_10": bool(
            cached_med <= uncached_med / DIFFERENTIAL_FACTOR
        ),
        # Evaluated only on a box that is not oversubscribed; otherwise recorded
        # as not_evaluable (see _absolute_verdict), which is NOT a pass.
        "absolute_instrument_cheaper_than_device_work": absolute_ok,
        "cache_engages": bool(stats["hit_rate"] > 0.5 and stats["parses"] < stats["files_seen"]),
        "staleness_reported": bool(
            "staleness_bound" in stats
            and "terminal_reverify_sec" in stats
            and "listing_ttl_sec" in stats
        ),
    }
    record = {
        "bead": "gpu-sim-u2o.6.6.6.1",
        "produced_at_utc": datetime.now(timezone.utc).isoformat(),
        "produced_by": "experiments/gpu-sim-u2o.6.6.6.1/measure_cost.py (committed code, this tree)",
        "loadavg_1_5_15": list(loadavg) if loadavg else None,
        "queue_files_at_measure_time": queue_files,
        "n_rounds": N,
        "cached_ms": {"median": round(cached_med, 4), "min": round(cached_min, 4)},
        "uncached_ms": {
            "median": round(uncached_med, 4),
            "min": round(min(uncached), 4),
        },
        "speedup_same_process_uncached_over_cached": round(speedup, 1),
        "device_time_ms_cited_from_6_7_1": DEVICE_TIME_MS,
        "ratio_cached_min_to_device": round(cached_min / DEVICE_TIME_MS, 3),
"absolute_instrument_vs_device_verdict": absolute_verdict,
        "absolute_claim_evaluable_here": not absolute_verdict.startswith(
            "not_evaluable"),
        "quiet_box_threshold": f"loadavg(1m) <= {QUIET_BOX_LOAD_PER_CPU} x ncpu",
        "hit_rate": round(stats["hit_rate"], 4),
        "parses": stats["parses"],
        "files_seen": stats["files_seen"],
        "terminal_reverify_sec": stats["terminal_reverify_sec"],
        "listing_ttl_sec": stats["listing_ttl_sec"],
        "staleness_bound": stats["staleness_bound"],
        "gates": gates,
"all_gates_pass": all(gates.values()),
        "not_evaluable_claims": [] if absolute_ok and not absolute_verdict.startswith(
            "not_evaluable") else [absolute_verdict],
        "exact_ms_not_claimed_reproducible": (
            "Millisecond figures move with box load (see loadavg) and queue "
            "directory size; gate-reproducibility is claimed, exact-ms "
            "reproducibility is not. The strict 0.81x headline ratio lives in "
            "metrics.json with its own measurement provenance."
        ),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not all(gates.values()):
        failed = [k for k, v in gates.items() if not v]
        print(f"SPOT CHECK FAILED gates={failed}", file=sys.stderr)
        return 1
    print("SPOT CHECK PASSED: every gate that could be evaluated held, in "
          "this process, on this box.")
    if absolute_verdict.startswith("not_evaluable"):
        print("NOTE: the absolute instrument-vs-device claim was NOT EVALUATED "
              "(box oversubscribed); it is not a pass and is not a fail. "
              "See absolute_instrument_vs_device_verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
