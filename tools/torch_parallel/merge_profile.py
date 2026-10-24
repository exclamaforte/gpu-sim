"""Merge torch.profiler traces (real kernels + Python stacks) with the interposer's
collective trace into ONE Perfetto timeline.

Two profiles of the same run, in two clock domains:

  torch.profiler   CLOCK_MONOTONIC (boot us)  kernels, aten ops, Python frames
  interposer       CLOCK_REALTIME  (epoch ns) the simulated collectives

train.py records both clocks at the same instant ("clock_anchor"), so the offset is
computed, not guessed. A guessed offset would slide every collective off the kernel
that issued it and the picture would look plausible while being wrong -- which is
the failure mode this project has hit repeatedly.

Layout per rank, three rows (tids are offset per rank so ranks never collide):

    rank N / python    Python frames, from torch.profiler with_stack
    rank N / kernels   CUDA kernels + aten ops + cuda runtime
    rank N / nccl      simulated collectives, from the interposer

The nccl row is deliberately separate from the kernel row: those slices are a
MODEL, not a measurement from the GPU, and mixing them into the kernel track would
make modelled time look measured.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

TID_STRIDE = 10
TRACK_PY, TRACK_KERNEL, TRACK_NCCL = 1, 2, 3


def _anchor_event(doc: dict) -> float | None:
    """Timestamp of the fg_clock_anchor marker inside a rank's torch trace.

    This is the ONLY safe way to line the two traces up: an event both sides
    timestamp, rather than an assumption about whose clock is whose. Measured
    example of why: torch ts 1.97e12 vs time.monotonic() 4.06e11 in one run.
    """
    for e in doc["traceEvents"]:
        if e.get("name") == "fg_clock_anchor" and e.get("ts") is not None:
            return float(e["ts"])
    return None


def _anchor(summary: Path) -> tuple[float, dict[int, dict]]:
    rows = [json.loads(l) for l in summary.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("empty summary")
    a = rows[0]["clock_anchor"]
    # realtime_us = real_ns/1000 ; monotonic_us = mono_us ; offset subtracts realtime
    offset_us = a["real_ns"] / 1000.0 - a["mono_us"]
    per_rank = {r["rank"]: r for r in rows}
    return offset_us, per_rank


real_at_anchor_ns = 0


def build(torch_files: list[Path], collective_rows: list[dict],
          offset_us: float, per_rank: dict[int, dict], label: str) -> dict:
    events: list[dict] = []
    kept = Counter()
    anchors: dict[int, float | None] = {}
    annotations: dict[int, list] = {}

    for f in torch_files:
        rank = int(f.name.rsplit("rank", 1)[1].split(".")[0])
        pid = rank + 1
        base_tid = rank * TID_STRIDE
        doc = json.loads(f.read_text())
        anchor_ts = _anchor_event(doc)
        xs = [e for e in doc["traceEvents"] if e.get("ph") == "X" and e.get("ts")]

        # torch.profiler mixes int and string tids; the category already decides
        # the row (python / kernels), so per-tid identity is not needed here.
        for e in xs:
            cat = e.get("cat", "")
            if cat == "python_function":
                tid = base_tid + TRACK_PY
            elif cat in ("kernel", "Kernel", "cuda_runtime", "cuda_driver", "cpu_op",
                         "gpu_memset", "user_annotation", "gpu_user_annotation"):
                tid = base_tid + TRACK_KERNEL
            else:
                continue
            out = {
                "ph": "X", "pid": pid, "tid": tid,
                "ts": e["ts"], "dur": max(e.get("dur") or 0.0, 0.05),
                "name": e.get("name", "?")[:200], "cat": cat,
            }
            if "stack" in e:
                out["args"] = {"stack": e["stack"]}
            elif e.get("args"):
                out["args"] = e["args"]
            events.append(out)
            kept[cat] += 1

        anchors[rank] = anchor_ts
        # torch's OWN record of each NCCL call, in order. The interposer knows the
        # call happened (and how long the fabric model says it took) but its
        # timestamp comes from a different clock AND a different instant inside the
        # call -- measured 404 us median offset against these annotations, i.e. a
        # slice sitting 40% of its own width away from the call it explains.
        # Pairing in order is exact because the sequence is deterministic.
        ann = sorted((e for e in xs
                      if str(e.get("name", "")).startswith("nccl:")),
                     key=lambda e: e["ts"])
        annotations[rank] = ann
        for name, tid in (("python", base_tid + TRACK_PY),
                          ("kernels", base_tid + TRACK_KERNEL),
                          ("nccl (modelled)", base_tid + TRACK_NCCL)):
            events.append({"ph": "M", "pid": pid, "tid": tid,
                           "name": "thread_name",
                           "args": {"name": f"rank {rank} / {name}"}})
        events.append({"ph": "M", "pid": pid, "name": "process_name",
                       "args": {"name": f"{label} rank {rank}"}})

    # Collective placement: anchor_ts + (real elapsed since the anchor).
    # delta uses the REALTIME clock only, on both ends, so no cross-clock math.
    nccl = 0
    paired = 0
    seen: Counter = Counter()
    by_rank: dict[int, list] = {}
    for r in collective_rows:
        by_rank.setdefault(r.get("grank", r.get("rank")), []).append(r)

    ordered: list[dict] = []
    for g, lst in by_rank.items():
        ann = annotations.get(g) or []
        for i, r in enumerate(lst):
            r = dict(r)
            if i < len(ann):
                r["_start_ts"] = float(ann[i]["ts"])
                paired += 1
            ordered.append(r)

    for r in ordered:
        g = r.get("grank", r.get("rank"))
        pred = r.get("predicted_s")
        dur = 1.0 if pred is None else max(float(pred) * 1e6, 0.05)
        a_ts = r.get("_start_ts", anchors.get(g))
        if a_ts is None:
            continue
        delta_us = 0.0 if "_start_ts" in r else (r["ts_ns"] - real_at_anchor_ns) / 1000.0
        events.append({
            "ph": "X", "pid": g + 1, "tid": g * TID_STRIDE + TRACK_NCCL,
            "ts": a_ts + delta_us, "dur": dur,
            "name": r.get("op", "collective"), "cat": "nccl",
            "args": {"bytes": r.get("bytes"), "dtype": r.get("dtype"),
                     "algorithm": r.get("algorithm"), "regime": r.get("regime"),
                     "nranks": r.get("nranks"),
                     "duration_modelled": pred is not None},
        })
        nccl += 1

    for rank, s in per_rank.items():
        start_ns = s.get("measure_start_ns")
        if not start_ns:
            continue
        acc = 0.0
        for i, dt in enumerate(s.get("step_times_s") or []):
            events.append({
                "ph": "i", "pid": rank + 1, "tid": rank * TID_STRIDE + TRACK_KERNEL,
                "ts": (start_ns / 1000.0) - offset_us + acc * 1e6,
                "name": f"step {i}", "s": "t",
            })
            acc += dt

    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "source": "torch.profiler (kernels, python) + libfakenccl JSONL (collectives)",
            "clock_alignment_us": offset_us,
            "counts": dict(kept) | {"nccl": nccl, "nccl_paired_to_annotation": paired},
            "CAVEAT": ("Kernels and Python frames are real, from ONE physical GPU "
                       "time-sliced across ranks. The nccl row is a MODEL on a "
                       "declared fabric, not GPU-measured -- it is a separate track "
                       "for that reason. Do not read the nccl row's widths as "
                       "hardware truth."),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-traces", required=True, help="glob, e.g. /tmp/prof2/p.rank*.json")
    ap.add_argument("--trace", required=True, help="interposer JSONL")
    ap.add_argument("--summary", required=True, help="train.py --json (needs clock_anchor)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="8xH100 declared")
    a = ap.parse_args(argv)

    files = sorted(Path(p) for p in glob.glob(a.torch_traces))
    if not files:
        raise SystemExit(f"no torch traces match {a.torch_traces}")
    rows = [json.loads(l) for l in Path(a.trace).read_text().splitlines() if l.strip()]
    offset_us, per_rank = _anchor(Path(a.summary))
    global real_at_anchor_ns

    global real_at_anchor_ns
    real_at_anchor_ns = per_rank[sorted(per_rank)[0]]["clock_anchor"]["real_ns"]
    doc = build(files, rows, offset_us, per_rank, a.label)
    Path(a.out).write_text(json.dumps(doc))
    print(json.dumps({"out": a.out, "ranks": len(files),
                      "events": len(doc["traceEvents"]),
                      "counts": doc["metadata"]["counts"],
                      "MB": round(Path(a.out).stat().st_size / 1e6, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
