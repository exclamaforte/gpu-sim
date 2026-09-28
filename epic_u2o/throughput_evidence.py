"""Real data-generation throughput, from committed evidence. gpu-sim-u2o.6.7.

gpu-sim-u2o.6.7 was reopened by the 6.15 audit because its backing artifact
was fabricated: runs/logs/throughput_demo_run.jsonl repeated the identical
"trials" [5.0, 5.1, 4.9] across its rows, hand-authored rather than measured.

This module exists so a throughput number can only be produced from evidence
that is actually there. The design rule, learned from the artifact it replaces:

    A THROUGHPUT NUMBER WITH NO UNDERLYING ROWS IS NOT A THROUGHPUT NUMBER.

So every function here takes the measured rows and derives; none accepts a
pre-computed rate, and a row without a real timestamp or a real trial set is an
error rather than a zero.

It also answers the question the bead was opened for -- is measurement
throughput the bottleneck? -- by separating the wall time the GPU spent running
kernels from the wall time the process spent NOT running them. On 7.1's real
batch that separation is roughly 10 000 : 1, which is the finding.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime


class ThroughputEvidenceError(RuntimeError):
    """Raised when a throughput number cannot be grounded in real rows."""


@dataclass
class ThroughputResult:
    n_rows: int
    wall_sec: float
    examples_per_sec: float
    examples_per_hour: float
    seconds_per_row: float
    derivation: str

    def to_dict(self) -> dict:
        return asdict(self)


def row_trials(row: dict) -> list:
    """The measured trial values of a row, under either schema in this lab.

    7.1 writes `invocation_medians_ms`; the tracker's own ExperimentLogger
    writes `trials`. Checking only one of them meant the fabricated artifact was
    refused for a SCHEMA MISMATCH -- "no trials found" -- while the check that
    actually matters, identical trials across every row, never ran. A guard that
    refuses for the wrong reason is a guard whose important path is untested.
    """
    for key in ("invocation_medians_ms", "trials"):
        v = row.get(key)
        if v:
            return list(v)
    return []


def _require_real_rows(rows: list) -> None:
    """Refuse a row set that cannot support a rate.

    The fabricated artifact failed here on three counts at once, and each check
    is cheap:
      * rows carrying no trials at all, yet marked completed;
      * every row carrying the IDENTICAL trial list, which is the signature of
        a hand-authored number;
      * rows with no real timestamp.
    """
    if not rows:
        raise ThroughputEvidenceError(
            "no rows: a throughput computed from nothing is not a throughput")

    empty = [r for r in rows if not row_trials(r)]
    if empty:
        raise ThroughputEvidenceError(
            f"{len(empty)} of {len(rows)} rows carry no measured trials. A row "
            f"with no measurements must not contribute to a rate, and marking "
            f"it 'completed' is how the fabricated artifact reported 15 "
            f"completed with 5 of them holding nothing.")

    distinct_trials = {json.dumps(row_trials(r), sort_keys=True)
                       for r in rows}
    if len(distinct_trials) == 1 and len(rows) > 1:
        raise ThroughputEvidenceError(
            f"all {len(rows)} rows carry the IDENTICAL trial set "
            f"{distinct_trials.pop()[:60]}. Measured runs do not repeat "
            f"themselves exactly; identical trials across every row is the "
            f"signature of a hand-authored number, not a slow one.")

    unstamped = [r for r in rows if not r.get("measured_at")]
    if unstamped:
        raise ThroughputEvidenceError(
            f"{len(unstamped)} rows carry no measured_at timestamp, so the "
            f"wall time cannot be derived from the rows themselves.")


def throughput_from_wall(rows: list, wall_sec: float, derivation: str) -> ThroughputResult:
    """Derive the rate from the row count and a wall clock.

    `wall_sec` must come from a real harness run. It is a parameter rather than
    a constant so a caller cannot accidentally ship a hardcoded rate, and the
    caller is expected to cross-check it with throughput_from_timestamps().
    """
    _require_real_rows(rows)
    if wall_sec <= 0:
        raise ThroughputEvidenceError(
            f"wall_sec={wall_sec} is not a duration; a zero or negative wall "
            f"clock produces an infinite rate, which is how a broken harness "
            f"turns into a headline number.")
    n = len(rows)
    return ThroughputResult(
        n_rows=n, wall_sec=wall_sec,
        examples_per_sec=n / wall_sec,
        examples_per_hour=(n / wall_sec) * 3600.0,
        seconds_per_row=wall_sec / n,
        derivation=derivation,
    )


def throughput_from_timestamps(rows: list) -> ThroughputResult:
    """Independent derivation: the span of the rows' own timestamps.

    This exists to CROSS-CHECK the harness's wall clock rather than replace it.
    A throughput number derived two ways that agree is evidence; one derived
    one way is an assertion.
    """
    _require_real_rows(rows)
    stamps = []
    for r in rows:
        try:
            stamps.append(datetime.strptime(r["measured_at"], "%Y-%m-%dT%H:%M:%S%z"))
        except (KeyError, ValueError) as exc:
            raise ThroughputEvidenceError(
                f"unparseable measured_at {r.get('measured_at')!r}: {exc}")
    span = (max(stamps) - min(stamps)).total_seconds()
    if span <= 0:
        raise ThroughputEvidenceError(
            "all rows share one timestamp, so the wall time is zero; the rows "
            "were not spread over a real run")
    return throughput_from_wall(
        rows, span, derivation="per-row measured_at span (independent of the "
                               "harness's own wall clock)")


def cross_check(a: ThroughputResult, b: ThroughputResult,
                tolerance_pct: float = 5.0) -> dict:
    """Two derivations that disagree mean one of them is wrong.

    Returning the comparison rather than a single number keeps the
    disagreement visible instead of averaging it away.
    """
    if a.examples_per_sec <= 0 or b.examples_per_sec <= 0:
        raise ThroughputEvidenceError("cannot cross-check a non-positive rate")
    delta = abs(a.examples_per_sec - b.examples_per_sec) / a.examples_per_sec * 100
    return {
        "examples_per_sec_harness_wall": a.examples_per_sec,
        "examples_per_sec_timestamp_span": b.examples_per_sec,
        "disagreement_pct": delta,
        "agree_within_tolerance": delta <= tolerance_pct,
        "tolerance_pct": tolerance_pct,
        "a_derivation": a.derivation,
        "b_derivation": b.derivation,
    }


def bottleneck_split(rows: list, wall_sec: float, launches_per_row: int) -> dict:
    """Separate GPU time from process overhead -- the bead's actual question.

    The bead was opened because measurement throughput, not training
    throughput, is expected to be this project's bottleneck. That is only
    answerable by asking what the wall time is SPENT on. Summing the measured
    kernel medians and subtracting from the wall gives the overhead: process
    spawn, driver init, and per-row setup.
    """
    _require_real_rows(rows)
    if launches_per_row <= 0:
        raise ThroughputEvidenceError("launches_per_row must be positive")

    gpu_sec = sum(r["median_runtime_ms"] for r in rows) / 1000.0
    total_launches = len(rows) * launches_per_row
    measured_kernel_sec = sum(
        sum(row_trials(r)) / 1000.0 for r in rows)
    overhead = wall_sec - measured_kernel_sec
    return {
        "wall_sec": wall_sec,
        "measured_kernel_sec": measured_kernel_sec,
        "overhead_sec": overhead,
        "overhead_fraction_of_wall": overhead / wall_sec,
        "kernel_fraction_of_wall": measured_kernel_sec / wall_sec,
        "n_rows": len(rows),
        "launches_per_row": launches_per_row,
        "total_launches": total_launches,
        "wall_sec_per_launch": wall_sec / total_launches,
        "kernel_us_per_launch_median": measured_kernel_sec / total_launches * 1e6,
        "verdict": (
            f"{overhead / wall_sec * 100:.2f}% of the measurement wall clock is "
            f"NOT the kernel running. The GPU is asked to do "
            f"{measured_kernel_sec / wall_sec * 100:.3f}% of the work and the "
            f"rest is process spawn, driver init and per-row setup. So the "
            f"bottleneck is per-EXAMPLE OVERHEAD, not GPU throughput and not "
            f"training throughput -- and the fix is batching or a persistent "
            f"worker, not a faster GPU."
            if measured_kernel_sec < wall_sec * 0.5 else
            "the wall clock is dominated by kernel execution; throughput is "
            "GPU-bound on this batch."),
    }
