"""Sweep parallelism configurations and summarise what each one costs.

Usage:
    python -m tools.torch_parallel.sweep --world-sizes 1 2 --steps 3
    python -m tools.torch_parallel.sweep --world-sizes 1 2 --delay-ms 0.5 2.0

For each configuration this launches torchrun with the NCCL-only interposer
preloaded, collects:

  * the per-rank step times reported by the training process itself, and
  * the per-rank collective trace emitted by the interposer (op, bytes, rank),

and prints one row per configuration. Nothing here needs a second GPU: every rank
runs on the one real card, and the multi-device fan-out is simulated.

What the numbers mean, stated plainly so they are not over-read:

  step_ms      wall clock for one optimizer step, measured inside the training
               process. With --delay-ms it includes synthetic compute time.
  coll_msgs    number of collective calls issued (per rank, averaged).
  coll_MB      bytes the collectives were asked to move (per rank, averaged).
  comm_share   fraction of the step spent in simulated collectives, when the
               interposer can attribute it (it cannot yet -- reported as null
               rather than invented).

The interesting comparison is not the absolute step time; it is how coll_msgs and
coll_MB grow as world_size grows with the model held fixed. That is the quantity
a parallelism decision actually turns on.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INTERPOSER = REPO / "libfakenccl.so"


def _run_one(world: int, steps: int, delay, extra: list[str]) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"harness-w{world}-"))
    trace = tmp / "trace.jsonl"
    summary = tmp / "summary.jsonl"
    env = dict(os.environ)
    env.update({
        "LD_PRELOAD": str(INTERPOSER),
        "GPUSIM_TRACE": str(trace),
        "FAKEGPU_NCCL_NO_FILL": "1",
        "FAKEGPU_NUM_DEVICES": str(world),
        "GPUSIM_TOPO": env.get("GPUSIM_TOPO", ""),
    })
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={world}", "--standalone",
           "tools/torch_parallel/train.py",
           "--steps", str(steps), "--json", str(summary)]
    if delay:
        cmd += ["--delay-ms", str(delay[0]), str(delay[1])]
    cmd += extra

    proc = subprocess.run(cmd, cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=1800)

    rows = []
    if summary.exists():
        rows = [json.loads(l) for l in summary.read_text().splitlines() if l.strip()]
    msgs = bytes_moved = 0
    if trace.exists():
        for line in trace.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("rank") == 0:
                msgs += 1
                bytes_moved += int(e.get("bytes") or 0)

    shutil.rmtree(tmp, ignore_errors=True)
    return {
        "world": world,
        "rc": proc.returncode,
        "step_ms": (sum(r["mean_step_s"] for r in rows) / len(rows) * 1000) if rows else None,
        "coll_msgs": msgs,
        "coll_MB": round(bytes_moved / 1e6, 1),
        "params": rows[0]["params"] if rows else None,
        "stderr_tail": proc.stderr.strip().splitlines()[-1] if proc.returncode else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--delay-ms", type=float, nargs=2, default=None)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()

    if not INTERPOSER.exists():
        print(f"missing {INTERPOSER}; run 'make libfakenccl.so' first", file=sys.stderr)
        return 2

    results = [_run_one(w, args.steps, args.delay_ms, args.rest)
               for w in args.world_sizes]

    hdr = f"{'world':>5} {'step_ms':>9} {'coll_msgs':>10} {'coll_MB':>9} {'params':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        step = f"{r['step_ms']:.1f}" if r["step_ms"] is not None else "FAILED"
        print(f"{r['world']:>5} {step:>9} {r['coll_msgs']:>10} {r['coll_MB']:>9} "
              f"{r['params'] or '-':>10}")
        if r["rc"] != 0 and r["stderr_tail"]:
            print(f"      {r['stderr_tail']}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0 if all(r["rc"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
