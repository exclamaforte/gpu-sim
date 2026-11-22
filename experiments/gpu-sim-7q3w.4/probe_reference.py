#!/usr/bin/env python3
"""A5 as restated: 8-way DDP vs a single-rank reference, same seed and data.

Increment 3 shows the eight ranks AGREE with each other.  Eight ranks can be
equally wrong, so this probe runs the same three AdamW steps twice:

  * 8-way: `probe_training.py` (8 DDP ranks, rank-sharded batches, gloo);
  * 1-way: one process that computes each of the same eight shard losses,
    averages those gradients (loss/NODES before backward), and steps AdamW.

Both start from `torch.manual_seed(1234)` and use the same per-(rank,step)
generators, so the comparison is an equivalence check, not a second
demonstration of agreement.  It reports the final loss on all eight shards,
the max-abs parameter delta, and the tolerance it asserts (1e-5 absolute).

The 8-way child saves rank0.pt through `--checkpoint-dir`; the reference reads
that exact checkpoint.  No NCCL data-path claim is made: the 8-way all_reduce
is gloo, a correctness control.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import torch
import torch.nn as nn
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from probe_training import NODES, STEPS, REQUIRED_MIB, mem_available_mib  # noqa: E402

BATCH = 4
SEQ = 32
D_MODEL = 512
TOL = 1e-5


def build_model():
    torch.manual_seed(1234)
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=8, dim_feedforward=2048, batch_first=True
        ),
        num_layers=4,
    )


def shard_batch(rank, step):
    gen = torch.Generator().manual_seed(2000 + rank * 100 + step)
    return torch.randn(BATCH, SEQ, D_MODEL, generator=gen)


def final_losses(model):
    """Mean loss over the same eight shards the 8-way ranks trained on."""
    losses = []
    with torch.no_grad():
        for rank in range(NODES):
            x = shard_batch(rank, STEPS - 1)
            losses.append(float(model(x).square().mean()))
    return losses


def reference_train():
    torch.set_num_threads(1)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(STEPS):
        opt.zero_grad()
        for rank in range(NODES):
            # The DDP all_reduce averages the eight gradients.  Dividing each
            # shard's loss by NODES before backward is the same average, summed
            # in rank order by one process.
            loss = model(shard_batch(rank, step)).square().mean()
            (loss / NODES).backward()
        opt.step()
    return model


def first_json(text):
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "checks" in obj:
            return obj
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    if args.steps != STEPS:
        print(f"A5 FAIL: this probe's reference is defined for {STEPS} steps", file=sys.stderr)
        return 2

    start_mem = mem_available_mib()
    if start_mem < REQUIRED_MIB:
        print(
            f"A5 REFUSE: MemAvailable {start_mem} MiB < required {REQUIRED_MIB} MiB; "
            f"no 8-way run launched. measured_start={start_mem} MiB"
        )
        return 2

    with tempfile.TemporaryDirectory(dir=str(REPO / ".lab" / "tmp")) as td:
        ckpt_dir = Path(td) / "ckpt"
        ckpt_dir.mkdir()
        proc = subprocess.run(
            [
                sys.executable,
                str(HERE / "probe_training.py"),
                "--checkpoint-dir",
                str(ckpt_dir),
                "--steps",
                str(STEPS),
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        result = first_json(proc.stdout)
        if proc.returncode != 0 or result is None:
            print(
                f"A5 FAIL: 8-way run rc={proc.returncode}; "
                "no single-rank comparison was made"
            )
            return 1
        eight_state = torch.load(ckpt_dir / "rank0.pt", map_location="cpu")
        eight_hashes = result.get("checkpoint_hashes", [])

    if len(set(eight_hashes)) != 1:
        print(f"A5 FAIL: 8-way checkpoint hashes are not equal: {eight_hashes}")
        return 1

    ref_model = reference_train()
    ref_state = ref_model.state_dict()
    eight_eval = final_losses(build_model_and_load(eight_state))
    ref_eval = final_losses(ref_model)
    eight_loss = sum(eight_eval) / len(eight_eval)
    ref_loss = sum(ref_eval) / len(ref_eval)
    loss_delta = abs(eight_loss - ref_loss)

    param_delta = 0.0
    worst_key = None
    for key in eight_state:
        delta = float((eight_state[key].float() - ref_state[key].float()).abs().max())
        if delta > param_delta:
            param_delta = delta
            worst_key = key

    checks = {
        "eight_way_run_ok": all(result.get("checks", {}).values()),
        "eight_checkpoint_hashes_equal": len(set(eight_hashes)) == 1,
        "one_way_reference_ran": True,
        "final_loss_within_tolerance": loss_delta <= TOL,
        "max_param_delta_within_tolerance": param_delta <= TOL,
    }
    out = {
        "checks": checks,
        "tolerance": {"absolute": TOL, "applies_to": "final mean loss and each parameter"},
        "eight_way_final_loss": eight_loss,
        "one_way_final_loss": ref_loss,
        "final_loss_delta": loss_delta,
        "max_abs_param_delta": param_delta,
        "worst_param": worst_key,
        "eight_way_eval_losses": eight_eval,
        "one_way_eval_losses": ref_eval,
        "eight_way_checkpoint_hashes": eight_hashes,
        "steps": STEPS,
        "n_shards": NODES,
        "mem_available_mib_start": start_mem,
    }
    print(json.dumps(out, indent=2))
    if all(checks.values()):
        print(
            f"A5 OK: 8-way_final_loss={eight_loss:.9f} "
            f"1-way_final_loss={ref_loss:.9f} loss_delta={loss_delta:.3e} "
            f"max_abs_param_delta={param_delta:.3e} tolerance={TOL:.0e} "
            f"worst_param={worst_key}"
        )
        return 0
    print(
        f"A5 FAIL: 8-way_final_loss={eight_loss:.9f} "
        f"1-way_final_loss={ref_loss:.9f} loss_delta={loss_delta:.3e} "
        f"max_abs_param_delta={param_delta:.3e} tolerance={TOL:.0e} "
        f"checks={checks}"
    )
    return 1


def build_model_and_load(state):
    model = build_model()
    model.load_state_dict(state)
    model.eval()
    return model


if __name__ == "__main__":
    sys.exit(main())
