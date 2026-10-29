"""Measure achieved HBM bandwidth on the local GPU (gpu-sim-3fog).

achieved_bw_scale in scripts/error_budget.py is asserted at 1.0 with provenance
"published-spec (datasheet, not measured here)". Its one-sided fragility is the
largest in the budget at the headline cell (N8_winner 37.17%, flip_down
0.814132), so a measured value below 81.4% of the datasheet spec would flip the
N=8 winner recommendation.

Method: streaming kernels over a working set larger than L2 so traffic reaches
DRAM; several repetitions; report median and spread, never a best-case shot.
Occupancy is recorded before and after because a busy GPU gives a wrong number.
"""
import json
import statistics
import subprocess
import time

import torch

GB = 1024 ** 3
WORKING_SET = 3 * GB          # >> 40 MB L2 on the 5090
REPS = 7


def neighbours():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return out or "none"
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable: {exc}"


def spec_bw_bytes_per_s():
    """Datasheet HBM bandwidth for the installed card, from nvidia-smi name."""
    name = torch.cuda.get_device_name(0)
    # RTX 5090: 1792 GB/s (32 GB GDDR7, 512-bit @ 28 Gbps).
    table = {"5090": 1792e9}
    for key, val in table.items():
        if key in name:
            return val, name
    raise SystemExit(f"no datasheet entry for {name!r}; add it rather than guessing")


def copy_kernel(n_elems):
    a = torch.empty(n_elems, dtype=torch.float32, device="cuda")
    a.uniform_()
    b = torch.empty_like(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    b.copy_(a)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, 2 * a.numel() * a.element_size()


def read_kernel(n_elems):
    a = torch.empty(n_elems, dtype=torch.float32, device="cuda")
    a.uniform_()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = a.sum()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, a.numel() * a.element_size()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    before = neighbours()
    spec, name = spec_bw_bytes_per_s()
    n_elems = WORKING_SET // 4

    # warm up allocation / clocks
    for _ in range(2):
        copy_kernel(n_elems // 8)

    results = {}
    for label, fn in (("copy", copy_kernel), ("read", read_kernel)):
        samples = []
        for _ in range(REPS):
            dt, nbytes = fn(n_elems)
            samples.append(nbytes / dt)
        results[label] = {
            "median_bytes_per_s": statistics.median(samples),
            "min_bytes_per_s": min(samples),
            "max_bytes_per_s": max(samples),
            "samples": samples,
        }

    after = neighbours()
    print(json.dumps({
        "device": name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "working_set_bytes": WORKING_SET,
        "repetitions": REPS,
        "spec_bytes_per_s": spec,
        "neighbours_before": before,
        "neighbours_after": after,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
