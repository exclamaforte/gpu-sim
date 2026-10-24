#!/usr/bin/env python3
"""gpu-sim-7o2: interleaved replicated timing for the case-study cells.

Runs the 4 case-study cells (d_model 512/1024 x {DP4, TP2+DP2}) round by round,
shuffling cell order within each round with a recorded seed, so shared-box
drift is sampled across configs instead of attributed to one. Same invocation
shape as the published table: world=4, 5 steps / 2 warmup, nvlink fabric,
transport off (timing-only default), step_ms = mean of per-rank means (exactly
sweep._run_one's metric -- traffic counts are NOT re-run, they are exact).

Usage:
  python3 experiments/gpu-sim-7o2/run_interleaved.py \
      --rounds 8 --seed 702 --round-offset 0 --out experiments/gpu-sim-7o2/reps_a.jsonl
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.torch_parallel.sweep import _run_one  # noqa: E402
from tools.torch_parallel.topology import emit  # noqa: E402

CELLS = [
    ("d512-DP4", 512, 1),
    ("d512-TP2DP2", 512, 2),
    ("d1024-DP4", 1024, 1),
    ("d1024-TP2DP2", 1024, 2),
]


def loadavg() -> str:
    try:
        return open("/proc/loadavg").read().strip()
    except OSError:
        return "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=702)
    ap.add_argument("--round-offset", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Advance past the shuffles belonging to earlier offsets so every job file
    # with the same seed draws the same global sequence (offsets partition it).
    orders = []
    for _ in range(args.round_offset + args.rounds):
        order = list(range(len(CELLS)))
        rng.shuffle(order)
        orders.append(order)
    orders = orders[args.round_offset:]

    topo_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "7o2-topo"
    topo_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    n_ok = n_fail = 0
    with open(out, "a") as f:
        for ri, order in enumerate(orders):
            for cell_idx in order:
                name, d, tp = CELLS[cell_idx]
                topo = str(emit(4, topo_dir / f"topo_r{args.round_offset + ri}.bin", "nvlink"))
                extra = ["--d-model", str(d)]
                if tp > 1:
                    extra += ["--tp", str(tp)]
                t0 = datetime.now(timezone.utc).isoformat()
                try:
                    r = _run_one(4, 5, None, extra, topo)
                    rc, sm = r.get("rc"), r.get("step_ms")
                except Exception as e:  # noqa: BLE001 -- a dead rep must not kill the job
                    rc, sm = "EXC", None
                    err = f"{type(e).__name__}: {e}"
                else:
                    err = "" if rc == 0 else r.get("stderr_tail", "")
                rec = {
                    "cell": name, "d_model": d, "tp": tp,
                    "round": args.round_offset + ri, "seed": args.seed,
                    "utc_start": t0,
                    "utc_end": datetime.now(timezone.utc).isoformat(),
                    "loadavg": loadavg(),
                    "rc": rc, "step_ms": sm,
                    "step_ms_hi": r.get("step_ms_hi") if rc == 0 else None,
                    "step_ms_lo": r.get("step_ms_lo") if rc == 0 else None,
                    "stderr_tail": err,
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(json.dumps(rec), flush=True)
                if rc == 0 and sm is not None:
                    n_ok += 1
                else:
                    n_fail += 1
    print(f"done: {n_ok} ok, {n_fail} failed -> {out}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
