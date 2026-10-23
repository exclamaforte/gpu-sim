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


def _run_one(world: int, steps: int, delay, extra: list[str],
             topology_path: str | None = None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"harness-w{world}-"))
    trace = tmp / "trace.jsonl"
    summary = tmp / "summary.jsonl"
    env = dict(os.environ)
    env.update({
        "LD_PRELOAD": str(INTERPOSER),
        "GPUSIM_TRACE": str(trace),
        "FAKEGPU_NCCL_NO_FILL": "1",
        "FAKEGPU_NUM_DEVICES": str(world),
        # The interposer reads GPUSIM_TOPOLOGY; without it collectives take the
        # fallback path and are traced with predicted_s=null (bytes, no cost).
        "GPUSIM_TOPOLOGY": topology_path or env.get("GPUSIM_TOPOLOGY", ""),
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
    p2p_msgs = p2p_bytes = 0
    pred_s = 0.0
    # Only collectives issued during the measured window, so the count and the
    # step times cover the same steps (see train.py's measure_start_ns).
    # PER RANK. Taking the first row's timestamp and applying it to every rank
    # made the count unstable: ranks enter their measured window at slightly
    # different instants, so one rank's warmup collectives leaked in on some runs
    # and not others -- observed as 33 vs 65 collectives for the same config.
    start_by_rank = {r["rank"]: r.get("measure_start_ns") for r in rows
                     if r.get("measure_start_ns")}
    pp_of_rank0 = 1
    if trace.exists():
        for line in trace.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            # grank is the WORLD rank; "rank" is communicator-local and repeats
            # across subgroups (every group has its own rank 0). Keying on the
            # local rank mixed distinct ranks' traffic into one bucket.
            gr = e.get("grank", e.get("rank"))
            rk_ns = start_by_rank.get(gr)
            if rk_ns is not None and e.get("ts_ns", 0) < rk_ns:
                continue
            if gr == 0:
                if e.get("op") in ("send", "recv", "sendrecv"):
                    p2p_msgs += 1
                    p2p_bytes += int(e.get("bytes") or 0)
                else:
                    msgs += 1
                    bytes_moved += int(e.get("bytes") or 0)
                pred_s += float(e.get("predicted_s") or 0.0)
    for r in rows:
        if r.get("rank") == 0:
            pp_of_rank0 = int(r.get("pp", 1))

    shutil.rmtree(tmp, ignore_errors=True)
    return {
        "world": world,
        "rc": proc.returncode,
        "step_ms": (sum(r["mean_step_s"] for r in rows) / len(rows) * 1000) if rows else None,
        # Slowest/fastest rank mean step: under PP the spread across stages is
        # where the pipeline bubble shows up (fill/drain idle on the stages).
        "step_ms_hi": (max(r["mean_step_s"] for r in rows) * 1000) if rows else None,
        "step_ms_lo": (min(r["mean_step_s"] for r in rows) * 1000) if rows else None,
        "coll_msgs": msgs,
        "coll_MB": round(bytes_moved / 1e6, 1),
        "p2p_msgs": p2p_msgs,
        "p2p_MB": round(p2p_bytes / 1e6, 1),
        "comm_sim_ms": round(pred_s * 1000, 2),
        "params": rows[0]["params"] if rows else None,
        "pp": pp_of_rank0,
        "stderr_tail": proc.stderr.strip().splitlines()[-1] if proc.returncode else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--delay-ms", type=float, nargs=2, default=None)
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel width; world becomes dp*tp so world must be a multiple")
    ap.add_argument("--pp", type=int, default=1,
                    help="pipeline-parallel width; world becomes pp*dp*tp so world must be a multiple")
    ap.add_argument("--micro-batches", type=int, default=2,
                    help="GPipe micro-batches per step (PP mode; must divide batch)")
    ap.add_argument("--fabric", default="nvlink",
                    help="declare a fabric so collectives carry predicted cost; '' to disable")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()

    if not INTERPOSER.exists():
        print(f"missing {INTERPOSER}; run 'make libfakenccl.so' first", file=sys.stderr)
        return 2

    from tools.torch_parallel.topology import emit
    tmp_topo = Path(tempfile.mkdtemp(prefix="harness-topo-"))
    results = []
    for w in args.world_sizes:
        topo = None
        if args.fabric:
            topo = str(emit(w, tmp_topo / f"topo_w{w}_{args.fabric}.bin", args.fabric))
        extra = list(args.rest)
        if args.tp > 1:
            extra += ["--tp", str(args.tp)]
        if args.pp > 1:
            extra += ["--pp", str(args.pp),
                      "--micro-batches", str(args.micro_batches)]
        results.append(_run_one(w, args.steps, args.delay_ms, extra, topo))
    shutil.rmtree(tmp_topo, ignore_errors=True)

    hdr = (f"{'world':>5} {'tp':>3} {'pp':>3} {'dp':>3} {'step_ms':>9} {'coll_msgs':>10} "
           f"{'coll_MB':>9} {'p2p_msgs':>9} {'p2p_MB':>8} {'comm_sim_ms':>12} {'params':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        step = f"{r['step_ms']:.1f}" if r["step_ms"] is not None else "FAILED"
        denom = args.tp * r["pp"]
        dp = r["world"] // denom if denom else r["world"]
        print(f"{r['world']:>5} {args.tp:>3} {r['pp']:>3} {dp:>3} {step:>9} {r['coll_msgs']:>10} "
              f"{r['coll_MB']:>9} {r['p2p_msgs']:>9} {r['p2p_MB']:>8} "
              f"{r['comm_sim_ms']:>12} {r['params'] or '-':>10}")
        if r["rc"] != 0 and r["stderr_tail"]:
            print(f"      {r['stderr_tail']}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0 if all(r["rc"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
