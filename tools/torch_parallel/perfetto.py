"""Convert the interposer's JSONL trace into a Perfetto / Chrome trace.

Perfetto (ui.perfetto.dev) reads Chrome trace JSON: an object with "traceEvents",
where each event either declares a track (metadata) or occupies time on it.

Our JSONL has one line per collective, per rank:

    {"ts_ns", "rank" (comm-local), "grank" (world), "nranks", "op", "bytes",
     "algorithm", "predicted_s", "regime", ...}

The mapping, and why each choice was made:

  track      one per grank. Comm-local "rank" repeats across subgroups (every
             group has its own rank 0), so it cannot identify a row -- this is the
             same defect class that hid a missing subgroup once already.
  slice      one X event per collective, starting at that rank's own ts_ns and
             lasting predicted_s. A collective with predicted_s null (refused by
             the fabric model) is drawn as a 1us marker rather than given an
             invented duration, and is flagged in args so the eye can find them.
  why-arg    args carry bytes/op/algorithm/regime so a click explains the slice.

WHAT THIS TRACE IS NOT: the compute on these rows happened on ONE physical GPU,
time-sliced across 8 processes, with a DECLARED 8xH100 fabric for the collectives.
The per-rank rows are the parallelism model's rows, not eight real devices. Anyone
reading the trace without that caveat would conclude the machine has eight H100s.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

US = 1_000_000  # ns -> us, Perfetto's native display unit


def convert(rows: list[dict], out: Path, label: str,
            step_marks: list[dict] | None = None) -> dict:
    ranks = sorted({r.get("grank", r.get("rank")) for r in rows})
    t0 = min(r["ts_ns"] for r in rows)

    events: list[dict] = []
    for g in ranks:
        events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": g + 1,
                       "args": {"name": f"rank {g}"}})
        events.append({"name": "process_name", "ph": "M", "pid": 1, "tid": g + 1,
                       "args": {"name": label}})

    refused = 0
    for r in rows:
        g = r.get("grank", r.get("rank"))
        pred = r.get("predicted_s")
        if pred is None:
            refused += 1
            dur_us = 1.0  # visible marker, not an invented cost
        else:
            dur_us = max(float(pred) * 1e6, 0.05)
        events.append({
            "ph": "X",
            "pid": 1,
            "tid": g + 1,
            "ts": (r["ts_ns"] - t0) / 1000.0,
            "dur": dur_us,
            "name": r.get("op", "collective"),
            "cat": r.get("algorithm", ""),
            "args": {
                "bytes": r.get("bytes"),
                "count": r.get("count"),
                "dtype": r.get("dtype"),
                "algorithm": r.get("algorithm"),
                "regime": r.get("regime"),
                "nranks": r.get("nranks"),
                "duration_modelled": pred is not None,
            },
        })

    # Per-step markers, when the harness summary is supplied: these give the trace
    # a top-level candidate for a training step the eye can navigate to.
    for s in (step_marks or []):
        events.append({
            "ph": "i", "pid": 1, "tid": s["rank"] + 1,
            "ts": s["ts_us"], "name": f"step {s['step']}", "s": "t",
            "args": {"note": "step boundary from the harness summary"},
        })

    doc = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "source": "libfakenccl.so JSONL trace",
            "ranks": ranks,
            "collectives": len(rows),
            "refused_unmodelled": refused,
            "CAVEAT": (
                "Compute ran on ONE physical GPU; the multi-device fan-out and all "
                "collective durations come from a declared fabric model. This trace "
                "describes a parallelism CONFIGURATION, not eight real accelerators."
            ),
        },
    }
    out.write_text(json.dumps(doc))
    return doc


def load_marks(summary_path: Path) -> list[dict]:
    """Step boundaries from the harness's per-rank JSON, in trace-relative time."""
    marks: list[dict] = []
    for line in summary_path.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        start_ns = s.get("measure_start_ns")
        if not start_ns:
            continue
        acc = 0.0
        for i, dt in enumerate(s.get("step_times_s") or []):
            marks.append({"rank": s["rank"], "step": i,
                          "ts_us": (start_ns - _t0_hint) / 1000.0 + acc * 1e6})
            acc += dt
    return marks


_t0_hint = 0


def main(argv=None) -> int:
    global _t0_hint
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="GPUSIM_TRACE jsonl")
    ap.add_argument("--out", required=True, help="perfetto trace json")
    ap.add_argument("--label", default="8xH100 (declared)")
    ap.add_argument("--summary", default=None, help="train.py --json output")
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(a.trace).read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("empty trace")

    marks = None
    if a.summary:
        _t0_hint = min(r["ts_ns"] for r in rows)
        marks = load_marks(Path(a.summary))

    doc = convert(rows, Path(a.out), a.label, marks)
    m = doc["metadata"]
    print(json.dumps({
        "out": a.out,
        "ranks": m["ranks"],
        "events": len(doc["traceEvents"]),
        "collectives": m["collectives"],
        "refused_unmodelled": m["refused_unmodelled"],
        "bytes": Path(a.out).stat().st_size,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
