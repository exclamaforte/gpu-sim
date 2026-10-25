#!/usr/bin/env python3
"""Model-vs-harness byte reconciliation (gpu-sim-wrvw).

Reads a harness trace (JSONL), sums bytes per op, normalises by steps*batch,
and compares against comm_bytes() from epic_cluster.parallelism_search.
Prints model/measured ratios; flags any outside [0.95, 1.05].
"""
import argparse, json, os, sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from epic_cluster import parallelism_search as ps


def read_trace(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--zero", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cp-grad-mode", type=str, default="production")
    # gpu-sim-wrvw: the comparison is meaningless unless the MODEL is the model the
    # harness actually ran. Without these the script compared llama3-8B's 16.6 GB of
    # parameter traffic against TinyLM's 2.7 MB and reported ratio 0.000 -- the
    # synthetic fixture matched only because it was generated from the same default.
    ap.add_argument("--model", choices=["llama3-8b", "tiny"], default="llama3-8b")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--seq", type=int, default=None)
    # the harness runs fp32 activations/grads; the model's baseline is bf16. Comparing
    # across dtypes silently doubles every ratio, so make it explicit.
    ap.add_argument("--dtype", choices=["bf16", "float32"], default="bf16")
    args = ap.parse_args()

    rows = read_trace(args.trace)
    meas = defaultdict(int)
    for r in rows:
        if r.get("rank", 0) != 0:
            continue
        op = r.get("op", "")
        meas[op] += int(r.get("bytes") or 0)

    norm = args.steps * args.batch if args.steps * args.batch > 0 else 1

    hw = {"peak_flops": 1e12, "mem_gb": 80.0, "fabric": "nvlink4_h100",
          "tp_payload": "four_d", "dp_dtype": args.dtype,
          "cp_grad_mode": args.cp_grad_mode}
    m = dict(ps.MODEL)
    if args.model == "tiny":
        # the harness's TinyLM (tools/torch_parallel/train.py defaults)
        m.update({"n_layers": 4, "d_model": 256, "n_heads": 4, "head_dim": 64,
                  "d_ff": 1024, "vocab": 4096, "seq": 512})
    for flag, key in (("layers", "n_layers"), ("d_model", "d_model"),
                      ("d_ff", "d_ff"), ("vocab", "vocab"), ("seq", "seq")):
        v = getattr(args, flag)
        if v is not None:
            m[key] = v
    if args.batch != 1:
        # the model prices ONE sequence per step; the harness runs --batch of them
        pass
    model = ps.comm_bytes(args.tp, args.pp, args.cp, args.dp, args.zero, hw, m)

    op_model = defaultdict(int)
    op_model["allreduce"] += model.get("tp_allreduce", 0)
    op_model["allreduce"] += model.get("dp_allreduce", 0)
    op_model["allgather"] += model.get("dp_allgather", 0)
    op_model["reducescatter"] += model.get("dp_reducescatter", 0)
    # cp_grad_sync: crude=allreduce, production=RS+AG
    cgs = model.get("cp_grad_sync", 0)
    if args.cp_grad_mode == "production":
        op_model["reducescatter"] += cgs // 2
        op_model["allgather"] += cgs // 2
    else:
        op_model["allreduce"] += cgs
    # cp_exchange and pp_sendrecv map to send/recv
    ce = model.get("cp_exchange", 0)
    psr = model.get("pp_sendrecv", 0)
    op_model["send"] += ce // 2 + psr // 2
    op_model["recv"] += ce // 2 + psr // 2

    print(f"{'op':<16} {'model_bytes':>14} {'meas_bytes':>14} {'ratio':>8}")
    print("-" * 56)
    for op in ["allreduce", "allgather", "reducescatter", "send", "recv"]:
        mb = op_model.get(op, 0)
        raw = meas.get(op, 0)
        meas_norm = raw // norm if norm > 0 else raw
        if op in ("allgather", "reducescatter") and mb > 0 and meas_norm > 0:
            # CP axis note (gpu-sim-juzb PI ruling): CP uses ring pricing in the
            # model but AG-based in the harness; cannot reconcile by construction.
            note = "  (cp axis: by-construction un-reconcilable, see gpu-sim-juzb)"
        else:
            note = ""
        if mb == 0 and meas_norm == 0:
            print(f"{op:<16} {mb:>14} {meas_norm:>14} {'n/a':>8}{note}")
            continue
        ratio = (meas_norm / mb) if mb > 0 else float("inf")
        flag = "" if (0.95 <= ratio <= 1.05) else "  <-- OUT OF RANGE"
        print(f"{op:<16} {mb:>14} {meas_norm:>14} {ratio:>8.3f}{flag}{note}")


if __name__ == "__main__":
    main()
