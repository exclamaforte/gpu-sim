"""gpu-sim-8yb4: price the K-window assumption in the corpus timings.

The corpus measures each config with ONE kernel launch between the two CUDA
events (epic_u2o/kernel_driver.py: WARMUP=5, TRIALS=20, launch-per-event-window).
The gate prices a kernel with K=40 launches per event window. If the per-launch
time is not flat in K, every predicted time inherits a fixed-cost error of
roughly (t_K1 - t_K40), and any comparison between corpus rows and gate prices is
priced in two different currencies.

This measures the same configs at K=1 and at K=40 IN THE SAME PROCESS FAMILY
(same synthesized source, same compiler flags, same box in the same minute) and
reports, per config: median per-launch time at K=1, at K=40, their ratio, and the
implied fixed cost per launch. Kernel-side time is divided by K by the generated
program itself, so both columns are per-launch microseconds.

Usage:
    python3 experiments/gpu-sim-8yb4/measure_k.py --out experiments/gpu-sim-8yb4/rows.jsonl
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from epic_u2o.generator_stage1 import generate_kernels      # noqa: E402
from epic_u2o.kernel_driver import synthesize               # noqa: E402

# (problem_size, block) -- grids follow from the launch geometry (grid = n/block),
# which is what the corpus used for the same sizes.
CONFIGS = [
    (8 * 1024 * 1024, 128),
    (8 * 1024 * 1024, 256),
    (32 * 1024 * 1024, 256),
    (32 * 1024 * 1024, 512),
]
K_TEST = 40


def spec_dict(n, block):
    grid = n // block
    specs = list(generate_kernels(ops=["scale"], grid_sweeps=[[grid, 1, 1]],
                                  block_sweeps=[[block, 1, 1]],
                                  problem_sizes=[n], max_kernels=1))
    if not specs:
        raise RuntimeError(f"generator produced no spec for n={n} block={block}")
    return specs[0].to_dict()


def _torch_includes():
    """The generated programs include torch/extension.h; derive the include dirs
    from the installed torch rather than assuming a venv is activated."""
    import torch
    inc = Path(torch.__file__).resolve().parent / "include"
    return [f"-I{inc}", f"-I{inc / 'torch' / 'csrc' / 'api' / 'include'}"]


def build(work: Path, spec: dict, k: int, arch: str) -> Path:
    cu = work / f"k{k}.cu"
    exe = work / f"k{k}.out"
    cu.write_text(synthesize(spec, k=k))
    r = subprocess.run(["nvcc", f"-arch={arch}", "-O2", *_torch_includes(),
                        str(cu), "-o", str(exe)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed for k={k}: {r.stderr[-600:]}")
    return exe


def measure(exe: Path):
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"{exe.name} exited {r.returncode}: {r.stderr[-400:]}")
    trials = [json.loads(line)["time_ms"] for line in r.stdout.splitlines()
              if line.startswith('{"trial"')]
    if len(trials) < 3:
        raise RuntimeError(f"{exe.name}: only {len(trials)} trials parsed")
    return statistics.median(trials), trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="sm_120")
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    work = Path(args.work) if args.work else out.parent / "build"
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, block in CONFIGS:
        spec = spec_dict(n, block)
        exe1 = build(work, spec, 1, args.arch)
        exeK = build(work, spec, K_TEST, args.arch)
        med1, trials1 = measure(exe1)
        medK, trialsK = measure(exeK)
        row = {
            "n": n,
            "block": block,
            "grid": n // block,
            "k1_median_us": round(med1 * 1000.0, 3),
            "k40_median_us": round(medK * 1000.0, 3),
            "ratio_k1_over_k40": round(med1 / medK, 4) if medK else None,
            "implied_fixed_cost_us": round((med1 - medK) * 1000.0, 3),
            "k1_trials_us": [round(t * 1000.0, 3) for t in trials1],
            "k40_trials_us": [round(t * 1000.0, 3) for t in trialsK],
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("n", "block", "k1_median_us",
                                             "k40_median_us", "ratio_k1_over_k40",
                                             "implied_fixed_cost_us")}), flush=True)

    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    ratios = [r["ratio_k1_over_k40"] for r in rows if r["ratio_k1_over_k40"]]
    fixed = [r["implied_fixed_cost_us"] for r in rows]
    summary = {
        "bead": "gpu-sim-8yb4",
        "question": ("Is per-launch kernel time flat in K? The corpus measures K=1 "
                     "launch per event window; the gate prices K=40."),
        "k_test": K_TEST,
        "configs": len(rows),
        "median_ratio_k1_over_k40": round(statistics.median(ratios), 4),
        "median_implied_fixed_cost_us": round(statistics.median(fixed), 3),
        "finding": ("per-launch time falls as K grows" if statistics.median(ratios) > 1.2
                    else "per-launch time is flat in K"),
        "rows_file": out.name,
    }
    (out.parent / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("MEASURE_K_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
