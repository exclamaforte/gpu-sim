#!/usr/bin/env python3
"""Increment (3): the 12.6M transformer trained by 8 ranks, equal checkpoints.

Eight ranks (one VDEV_NODE each) join a gloo process group, build the same
12,609,536-parameter transformer (d_model=512, 4 layers, ff=2048), wrap it in
DistributedDataParallel, and take real AdamW steps on rank-sharded synthetic
data.  Each rank prints its loss at every step and a canonical SHA-256 of its
final state_dict; the hashes must be EQUAL across all eight ranks.  A hash
disagreement is printed with the per-rank values and is a failure, except in
--inject-checkpoint-difference, where rank 3 perturbs one parameter and the
control succeeds only if the checker SEES the difference.

The training math is CPU float32 on purpose: gloo is the control plane here,
and CPU determinism is what makes a rank-equal checkpoint hash a meaningful
test of the distributed step.  The GPU identity is A1's measured claim; the
NCCL data path is 7q3w.3/7q3w.5 and is not claimed by this probe.

    python3 experiments/gpu-sim-7q3w.4/probe_training.py
    python3 experiments/gpu-sim-7q3w.4/probe_training.py --inject-checkpoint-difference
"""
import argparse
import json
import os
import contextlib
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_vdev import BUILD, REPO, build  # noqa: E402
from checkpoint_agreement import check_checkpoint_hashes  # noqa: E402

NODES = 8
STEPS = 3
BATCH = 4
SEQ = 32
D_MODEL = 512
# Measured with a 2-rank DDP run of this model+AdamW: 1191/1194 MiB.  Eight
# ranks are the design's shape; the margin is for import/allocator jitter and
# the parent.  This is a floor, and the measured per-rank RSS is reported.
PER_RANK_MIB = 1194
MARGIN_MIB = 1024
REQUIRED_MIB = NODES * PER_RANK_MIB + MARGIN_MIB

CHILD = r'''
import json, os, resource, sys
import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, os.environ["EXP_DIR"])
from checkpoint_agreement import canonical_state_dict_sha256

torch.set_num_threads(1)

node = int(os.environ["VDEV_NODE"])
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
steps = int(os.environ["TRAIN_STEPS"])
ckpt_dir = os.environ["CKPT_DIR"]
out = {"node": node, "rank": rank, "world_size": world_size, "steps": steps}
out["device_count"] = torch.cuda.device_count()
out["names"] = [torch.cuda.get_device_name(i) for i in range(out["device_count"])]

dist.init_process_group(backend="gloo", init_method="env://")
out["pg_initialized"] = dist.is_initialized()
out["pg_world_size"] = dist.get_world_size()
out["pg_rank"] = dist.get_rank()

# Identical initial weights on every rank.  DDP also broadcasts from rank 0,
# so a divergence here would be visible rather than silently averaged away.
torch.manual_seed(1234)
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
    ),
    num_layers=4,
)
ddp = nn.parallel.DistributedDataParallel(model)
opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

losses = []
for step in range(steps):
    # Rank-sharded synthetic data: every rank trains on different tokens, so
    # equal checkpoints are a product of the all-reduce, not identical inputs.
    gen = torch.Generator().manual_seed(2000 + rank * 100 + step)
    x = torch.randn(4, 32, 512, generator=gen)
    loss = ddp(x).square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
out["losses"] = losses

if os.environ.get("INJECT_CHECKPOINT_DIFFERENCE") == "1" and rank == 3:
    with torch.no_grad():
        next(model.parameters()).add_(0.01)

# One canonical hash, shared with the ledger-bound test (checkpoint_agreement.py):
# key order, dtype, shape and CPU tensor bytes; container metadata is excluded.
sd = model.state_dict()
out["checkpoint_sha256"] = canonical_state_dict_sha256(sd)
out["params"] = sum(p.numel() for p in model.parameters())

torch.save(sd, os.path.join(ckpt_dir, f"rank{rank}.pt"))
out["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
with open("/proc/meminfo") as f:
    for line in f:
        if line.startswith("MemAvailable:"):
            out["mem_available_mib"] = int(line.split()[1]) // 1024

dist.barrier()
dist.destroy_process_group()
print("TRAIN " + json.dumps(out))
'''


def mem_available_mib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not in /proc/meminfo")


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=NODES)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--inject-checkpoint-difference", action="store_true")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="persistent checkpoint directory (A5 reference needs rank0.pt)")
    ap.add_argument("--force-refuse-mib", type=int, default=None,
                    help="forced measured start for the REFUSE control (no ranks launched)")
    args = ap.parse_args()

    start_mem = (args.force_refuse_mib
                 if args.force_refuse_mib is not None else mem_available_mib())
    if start_mem < REQUIRED_MIB:
        forced = " (forced control)" if args.force_refuse_mib is not None else ""
        print(
            f"REFUSE: MemAvailable {start_mem} MiB < required {REQUIRED_MIB} MiB "
            f"({args.nodes}*{PER_RANK_MIB} + {MARGIN_MIB} margin); "
            f"no rank launched. measured_start={start_mem} MiB{forced}"
        )
        return 2

    build()
    port = free_port()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(BUILD) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["LD_PRELOAD"] = (str(BUILD / "libnvidia-ml.so.1") + ":"
                        + str(BUILD / "libcudart-vdev.so.1"))
    env["VDEV_NVDEV"] = "8"
    env["EXP_DIR"] = str(HERE)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    env["WORLD_SIZE"] = str(args.nodes)
    env["TRAIN_STEPS"] = str(args.steps)
    env["OMP_NUM_THREADS"] = "1"
    if args.inject_checkpoint_difference:
        env["INJECT_CHECKPOINT_DIFFERENCE"] = "1"

    scratch = (contextlib.nullcontext(str(args.checkpoint_dir.parent))
               if args.checkpoint_dir else
               tempfile.TemporaryDirectory(dir=str(REPO / ".lab" / "tmp")))
    with scratch as td:
        if args.checkpoint_dir:
            ckpt_dir = args.checkpoint_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        else:
            ckpt_dir = Path(td) / "checkpoints"
            ckpt_dir.mkdir()
        env["CKPT_DIR"] = str(ckpt_dir)
        procs = []
        for n in range(args.nodes):
            e = dict(env)
            e["VDEV_NODE"] = str(n)
            e["RANK"] = str(n)
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", CHILD],
                    env=e,
                    cwd=td,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        outputs = []
        failures = []
        for n, p in enumerate(procs):
            out, err = p.communicate(timeout=1800)
            line = [ln for ln in out.splitlines() if ln.startswith("TRAIN ")]
            if not line:
                failures.append(
                    f"rank {n}: rc={p.returncode}\n{out[-1200:]}\n{err[-1200:]}"
                )
                continue
            outputs.append(json.loads(line[0][6:]))
        ckpt_files = sorted(p.name for p in ckpt_dir.glob("*.pt"))

    end_mem = mem_available_mib()
    if failures or len(outputs) != args.nodes:
        print("\n".join(failures))
        print(
            f"FAIL: {len(outputs)}/{args.nodes} ranks reported; "
            f"mem_available_start={start_mem} end={end_mem}"
        )
        return 1

    per_rank = {}
    for o in outputs:
        per_rank[o["rank"]] = {
            "node": o["node"],
            "device_count": o["device_count"],
            "names": o["names"],
            "params": o["params"],
            "losses": o["losses"],
            "checkpoint_sha256": o["checkpoint_sha256"],
            "peak_rss_mib": o["peak_rss_mib"],
            "mem_available_mib": o.get("mem_available_mib"),
        }

    hashes = [r["checkpoint_sha256"] for r in per_rank.values()]
    agreement = check_checkpoint_hashes(hashes)
    equal_hash = agreement["agrees"]
    checks = {
        "eight_ranks": len(per_rank) == args.nodes,
        "each_rank_is_its_node": all(
            r["node"] == rank for rank, r in per_rank.items()
        ),
        "process_group_world_size_8": all(
            o["pg_world_size"] == args.nodes and o["pg_rank"] == o["rank"]
            and o["pg_initialized"]
            for o in outputs
        ),
        "every_rank_sees_8": all(r["device_count"] == 8 for r in per_rank.values()),
        "every_rank_names_b300": all(
            all("B300" in x for x in r["names"]) for r in per_rank.values()
        ),
        "model_is_12_6M_params": all(r["params"] == 12609536 for r in per_rank.values()),
        "three_losses_per_rank": all(len(r["losses"]) == args.steps for r in per_rank.values()),
        "losses_finite": all(
            all(isinstance(x, float) and x == x and abs(x) != float("inf")
                for x in r["losses"])
            for r in per_rank.values()
        ),
        "checkpoint_hash_equal": equal_hash,
        "all_checkpoint_files_saved": len(ckpt_files) == args.nodes,
    }

    result = {
        "per_rank": per_rank,
        "checkpoint_hashes": hashes,
        "checkpoint_agreement": agreement,
        "checkpoints": ckpt_files,
        "checks": checks,
        "rss_sum_mib": sum(r["peak_rss_mib"] for r in per_rank.values()),
        "min_child_mem_available_mib": min(
            r["mem_available_mib"] for r in per_rank.values()
        ),
        "mem_available_mib_start": start_mem,
        "mem_available_mib_end": end_mem,
        "preflight_required_mib": REQUIRED_MIB,
        "injected_checkpoint_difference": args.inject_checkpoint_difference,
    }
    result["checkpoint_dir"] = str(ckpt_dir)
    print(json.dumps(result, indent=2))

    if args.inject_checkpoint_difference:
        detected = not equal_hash and len(set(hashes)) > 1
        other_ok = all(v for k, v in checks.items() if k != "checkpoint_hash_equal")
        if detected and other_ok:
            print(
                "CHECKPOINT DISAGREEMENT DETECTED (control): per-rank hashes "
                f"{hashes}; the equality check is fail-able."
            )
            return 0
        print(
            "FAIL(control): injected checkpoint difference was not detected; "
            f"hashes={hashes} checks={checks}"
        )
        return 1

    if not all(checks.values()):
        print("FAIL: " + ", ".join(k for k, v in checks.items() if not v))
        print(f"CHECKPOINT HASHES: {sorted(set(hashes))}")
        print("A checkpoint-hash disagreement is the result; it is not being fixed silently.")
        return 1
    print(
        f"TRAIN OK: {args.nodes} ranks x 12,609,536 params, {args.steps} real AdamW steps, "
        f"rank-sharded data, checkpoint_sha256={hashes[0]} equal on all "
        f"{args.nodes} ranks; rss_sum={result['rss_sum_mib']} MiB, "
        f"mem_available_start={start_mem} MiB end={end_mem} MiB "
        f"preflight_required={REQUIRED_MIB} MiB"
    )
    for rank in sorted(per_rank):
        print(f"  rank {rank}: losses={per_rank[rank]['losses']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
