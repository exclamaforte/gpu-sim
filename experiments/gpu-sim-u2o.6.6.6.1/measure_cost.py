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

Checks. Every gate is DIFFERENTIAL, so box state cancels rather than deciding
the verdict:

  1. differential (the load-bearing guard): warm-cached median <= uncached
     median / 10, both arms measured back-to-back in ONE PROCESS. A
     reintroduced full scan measures ~1x by construction and fails at any
     load. This is the guard.
  2. the cache engages: hit_rate > 0.5 and parses < files_seen.
  3. the staleness bound is reported: load_stats carries staleness_bound,
     terminal_reverify_sec, listing_ttl_sec.

The absolute instrument-vs-device comparison is REPORTED, never gated. Three
attempts to gate it were made and all three were wrong, and the reasons are
the point:

  * a live wall-clock median vs a hardcoded 0.139 ms -- failed under load with
    no code change;
  * the same with a 2x margin -- failed at 0.279 ms against a 0.278 ms
    threshold at loadavg 94, with a repair (raise the constant) that reinstates
    the hand-edited number;
  * conditioning on loadavg(1m) <= ncpu -- at loadavg 14.3, "quiet" by that
    standard, the instrument still measured 1.91x. The condition did not
    exclude the failing regime, which makes it worse than no gate because it
    LOOKS calibrated.

NO TIMING CONSTANT IS BAKED INTO THIS FILE. The marginal cost per queue file
is load-dependent -- ~0.185 us/file at loadavg 4.3 and ~0.28 at loadavg 38 on
this host, a ~1.5x spread -- so the slope and the crossover are DERIVED PER RUN
by measure_size_sweep() and reported next to the loadavg they were measured
at. An earlier version hardcoded "~0.188 us/file" and "crossover ~739", which
described one machine at one moment while reading as a property of the code;
that is the defect the reviewer caught.

The strict 0.81x headline in metrics.json is the artifact measurement at its
own reference size (557 files) and is not re-asserted per run here.

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

DEVICE_TIME_MS = 0.139  # gpu-sim-u2o.6.7.1 measured device work per example
DIFFERENTIAL_FACTOR = 10.0
N = 15

# The criterion-1 claim is DERIVED PER RUN, never asserted from a constant.
#
# Every timing figure this file reports comes from measure_size_sweep(), run in
# this process, and is stamped with the loadavg measured at the same moment. The
# hardcoded numbers that used to live here -- the ~0.188 us/file slope, the 819
# live-file count, the ~739-file crossover, a ~1.3x load factor -- were removed
# because each was one machine at one moment written as a property of the code.
# See _scaling_law() and the module docstring for what is asserted, and for why
# two earlier attempts to gate on an absolute or per-file millisecond figure were
# both wrong.
REFERENCE_DIR_FILES = 557          # the directory size the artifact measured
# REFERENCE_DIR_FILES is the size the committed artifact measured at, kept so the
# artifact's numbers stay interpretable. It is NOT a threshold and NOTHING is
# gated against it: two attempts to gate on an absolute or per-file millisecond
# figure were both wrong -- a 2x margin failed at 0.279 vs 0.278 ms at loadavg
# 94, and a loadavg(1m)<=ncpu condition still passed at 1.91x when loadavg was
# 14.3. Load moves the per-file cost by ~1.5x on its own (measured: ~0.185
# us/file at loadavg 4.3 against ~0.28 at loadavg 38), so no fixed constant
# separates the regimes. The gates are therefore all DIFFERENTIAL -- see
# _reference_verdict and the module docstring.


def _scaling_law(points, loadavg_1m, slope_us_per_file) -> str:
    """The measured relationship, DERIVED PER RUN and stamped with its load.

    Nothing here is a constant baked into the file. The marginal cost is
    load-dependent (~0.185 us/file at loadavg 4.3, ~0.28 at loadavg 38 on this
    host), so quoting a bare number describes one machine at one moment.
    """
    if slope_us_per_file is None:
        return ("size sweep unavailable this run; no scaling claim is made.")
    crossover = DEVICE_TIME_MS / (slope_us_per_file / 1e3)
    return (f"measured THIS RUN at loadavg(1m)={loadavg_1m:.1f}: cost is linear "
            f"in queue directory size at {slope_us_per_file:.4f} us/file "
            f"(sweep {[(n, round(m, 4)) for n, m in points]}); crossover vs the "
            f"{DEVICE_TIME_MS} ms device budget is ~{crossover:.0f} files AT THIS "
            f"LOAD. Both figures move with load and with the machine.")



# ---------------------------------------------------------------------------
# gpu-sim-u2o.6.6.6.1 rework 4: the 6.6.6 contract break.
#
# After gpu-sim-u2o.6.6 merged, queue_state() with no identifying argument
# RAISES IdentificationError instead of returning a record with own_job_id=null
# -- 6.6.6's fail-loudly property, and it is correct. It also breaks every bare
# call site in this file, because this file is an INSTRUMENT SELF-TEST running
# outside the queue, where there IS no own job to identify.
#
# The wrong fix is to catch the exception and carry on, which would rebuild the
# null own_job_id that 6.6.6 exists to remove. The wrong fix on the other side
# is to invent an identifying argument that happens to resolve.
#
# So the helper below states what the call IS: a cost measurement of the loader,
# not an audit record. It attempts identification, lets it be REFUSED, and
# records the refusal as a field rather than swallowing it. An absent or refused
# field must never read as coverage.
# ---------------------------------------------------------------------------


def benchmark_call(**kwargs):
    """queue_state() for cost measurement. Returns (record, identification).

    `identification` is one of "resolved", "refused", "error:<Type>". A refused
    identification is NOT a pass and NOT a null own_job_id -- it is recorded, so
    a reader of any artifact produced here can see that this record makes no
    claim about which job was ours, because there was none.
    """
    try:
        return qs.queue_state(**kwargs), "resolved"
    except qs.IdentificationError as exc:
        return {"own_job_id": None,
                "identification": "refused",
                "identification_error": str(exc)[:200],
                "scope": ("instrument self-test outside the queue: this record "
                          "makes NO claim about our own job, which is correct "
                          "because there is none. It is not an audit record.")}, \
               "refused"
    except Exception as exc:                       # never hide an unexpected break
        return {"own_job_id": None, "identification": f"error:{type(exc).__name__}",
                "error": str(exc)[:200]}, f"error:{type(exc).__name__}"

def measure_size_sweep(sizes=(200, 400, 600, 800), trials: int = 7) -> list:
    """Median warm cost of the instrument against TEMPORARY dirs of n files.

    Returned points are (n_files, median_ms). Every number derived from these
    points is DERIVED PER RUN and reported together with the load average
    measured in the same run -- never as a constant baked into this file.

    Why per-run and not a constant: the marginal cost is load-dependent. Measured
    on this host, ~0.185 us/file at loadavg 4.3 and ~0.28 us/file at loadavg 38,
    a ~1.5x spread. A hardcoded slope describes one machine at one moment, which
    is how "0.188 us/file" and "crossover ~739" ended up written as properties of
    the code when they are properties of a load. Touches throwaway directories
    only, never the shared queue.
    """
    import json as _json
    import shutil as _shutil
    import tempfile as _tempfile
    from pathlib import Path as _Path
    import importlib.util as _ilu

    mod_path = _Path(__file__).resolve().parents[2] / "epic_u2o" / "queue_state.py"
    points = []
    for n in sizes:
        tmp = _Path(_tempfile.mkdtemp(prefix="qs_size_"))
        try:
            for i in range(n):
                (tmp / f"job{i:05d}.json").write_text(_json.dumps({
                    "id": f"j{i}", "state": "finished",
                    "command": ["python3", "x.py"], "description": "sweep",
                }))
            spec = _ilu.spec_from_file_location("qs_size_arm", mod_path)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.QUEUE_DIR = tmp
            # Since 6.6.6 merged, queue_state() refuses to write a record
            # with a null own_job_id. A synthetic queue has no own job, so the
            # arm must IDENTIFY one of the jobs it just wrote. Previously it
            # called bare, every point raised, and the sweep returned [] --
            # which SPOT CHECK PASS then reported as success. A silent empty
            # measurement is worse than a loud one.
            ours = f"job{n - 1:05d}"
            mod.queue_state(own_job_id=ours)       # warm
            samples = []
            for _ in range(trials):
                t0 = time.perf_counter()
                mod.queue_state(own_job_id=ours)
                samples.append((time.perf_counter() - t0) * 1e3)
            points.append((n, statistics.median(samples)))
        except Exception as exc:                    # never break the run
            print(f"size-sweep arm n={n} failed: {exc}", file=sys.stderr)
        finally:
            _shutil.rmtree(tmp, ignore_errors=True)
    return points


def fit_slope_us_per_file(points: list) -> float | None:
    """Least-squares slope of ms against n_files, in microseconds per file."""
    pts = [(n, ms) for n, ms in points if n > 0]
    if len(pts) < 2:
        return None
    xs = [n for n, _ in pts]
    ys = [ms for _, ms in pts]
    denom = sum(x * x for x in xs)
    if denom == 0:
        return None
    return (sum(x * y for x, y in zip(xs, ys)) / denom) * 1e3

def single_cached_call_ms() -> float:
    t0 = time.perf_counter()
    benchmark_call()
    return (time.perf_counter() - t0) * 1e3


def single_uncached_call_ms() -> float:
    qs.reset_cache()  # cold: nothing cached, full glob + parse on next call
    t0 = time.perf_counter()
    benchmark_call()
    return (time.perf_counter() - t0) * 1e3


def main() -> int:
    # Interleave the arms round by round so a load drift mid-run hits both.
    qs.reset_cache()
    benchmark_call()   # warm once; first-call cost is not the steady state
    cached, uncached = [], []
    for _ in range(N):
        cached.append(single_cached_call_ms())
        uncached.append(single_uncached_call_ms())
        benchmark_call()   # re-warm after the reset for the next cached sample
    qs.reset_cache()
    for _ in range(5):  # re-warm, then collect steady-state stats for the record
        benchmark_call()
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

    n_files = queue_files or 0
    # Derived THIS RUN, never a constant: the marginal cost is load-dependent.
    sweep_points = measure_size_sweep()
    slope_us = fit_slope_us_per_file(sweep_points)
    load1 = loadavg[0] if loadavg else float("nan")
    crossover_n = (DEVICE_TIME_MS / (slope_us / 1e3)) if slope_us else None
    gates = {
        # A sweep that produced no points is not a pass with nothing to say --
        # it is a measurement that did not happen. Before this gate existed the
        # rebase's contract break made every sweep point raise, the sweep came
        # back empty, and SPOT CHECK PASS reported success anyway.
        "size_sweep_produced_points": bool(sweep_points),
        "differential_cached_le_uncached_over_10": bool(
            cached_med <= uncached_med / DIFFERENTIAL_FACTOR
        ),
# NOT a gate. The absolute instrument-vs-device comparison is REPORTED,
        # not gated, because it is currently FALSE at live size and a gate that
        # passed would be misleading. See criterion_1_at_live_size below and
        # gpu-sim-u2o.6.6.6.3 for the structural follow-on.
        "cache_engages": bool(stats["hit_rate"] > 0.5 and stats["parses"] < stats["files_seen"]),
        "staleness_reported": bool(
            "staleness_bound" in stats
            and "terminal_reverify_sec" in stats
            and "listing_ttl_sec" in stats
        ),
    }
    # Two-valued gates only: there is no None left to fold into a pass.
    failed = [k for k, v in gates.items() if v is False]
    gate_summary = "fail" if failed else "pass"
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
"ratio_cached_median_to_device_at_live_size": round(
            cached_med / DEVICE_TIME_MS, 3),
        "reference_dir_files": REFERENCE_DIR_FILES,
        "per_file_cost_us": (round(cached_med * 1e3 / n_files, 5)
                             if n_files else None),
        "size_sweep_points_this_run": [[n, round(m, 5)] for n, m in sweep_points],
        "slope_us_per_file_this_run": (round(slope_us, 5) if slope_us else None),
        "crossover_files_this_run": (round(crossover_n) if crossover_n else None),
        "loadavg_1m_at_sweep": round(load1, 2) if loadavg else None,
        "scaling_law": _scaling_law(sweep_points, load1, slope_us),
        "criterion_1_at_live_size": {
            "met": bool(cached_med < DEVICE_TIME_MS),
            "ratio": round(cached_med / DEVICE_TIME_MS, 3),
            "ratio_is_load_dependent": True,
            "loadavg_1m": round(load1, 2) if loadavg else None,
            "live_queue_files": n_files,
            "crossover_files_this_run": (round(crossover_n) if crossover_n else None),
            "statement": (
                "The committed artifact measured criterion 1 as MET at 557 queue "
                "files (0.1123 ms = 0.81x of the 0.139 ms device work). The live "
                "queue directory is now larger. The residual sweep is O(n) even "
                "fully cached -- the ACTIVE set is re-read and re-parsed on every "
                "call by design -- so a growing queue costs more per call. Nothing "
                "regressed; the input grew past the crossover. THIS RATIO IS "
                "LOAD-DEPENDENT and is stamped with the loadavg measured in the "
                "same run: marginal cost was ~0.185 us/file at loadavg 4.3 and "
                "~0.28 us/file at loadavg 38 on this host, a ~1.5x spread. Quoting "
                "it without its load describes one machine at one moment. This run "
                "REPORTS the live figure rather than gating it: a gate that passed "
                "here would be misleading and one that failed would redden run.sh "
                "on box state."),
        },
        "hit_rate": round(stats["hit_rate"], 4),
        "parses": stats["parses"],
        "files_seen": stats["files_seen"],
        "terminal_reverify_sec": stats["terminal_reverify_sec"],
        "listing_ttl_sec": stats["listing_ttl_sec"],
        "staleness_bound": stats["staleness_bound"],
        "gates": gates,
        "gate_summary": gate_summary,
        "gates_failed": sorted(failed),
        "exact_ms_not_claimed_reproducible": (
            "Millisecond figures move with box load (see loadavg) and queue "
            "directory size; gate-reproducibility is claimed, exact-ms "
            "reproducibility is not. The strict 0.81x headline ratio lives in "
            "metrics.json with its own measurement provenance."
        ),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"SPOT CHECK {gate_summary.upper()}: "
          + (f"failed gates {sorted(failed)}" if failed else "no gate failed"))
    print(f"SIZE SWEEP (derived this run at loadavg(1m)={load1:.1f}): "
          f"{[(n, round(m, 4)) for n, m in sweep_points]}")
    print(f"  marginal cost {slope_us:.4f} us/file; crossover "
          f"~{crossover_n:.0f} files AT THIS LOAD" if slope_us
          else "  size sweep unavailable; no scaling claim made")
    print(f"CRITERION 1 AT LIVE SIZE (reported, never gating): "
          f"{cached_med / DEVICE_TIME_MS:.2f}x device time at {n_files} live "
          f"queue files, at loadavg(1m)={load1:.1f}. Ratio is load-dependent.")
    # Neither box state nor queue growth may decide the exit code.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
