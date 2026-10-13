#!/usr/bin/env python3
"""First real batch runner (Phase-1, gated).

BUILD (CPU-only, run directly): generate stage-1 specs, synthesize drivers,
nvcc-compile each to a standalone binary, write a manifest. No GPU contact.

RUN (GPU-claiming, queue via `ts` ONLY): for each manifest entry, execute
the prebuilt binary under the harness timeout, parse per-trial JSON,
compute median/IQR via measurement_stats, and append real rows
(kernel_hash, launch_config, median_runtime_ms, iqr_ms,
device_prop_snapshot) to runs/stage1_dataset_measured/batch.jsonl.

The RUN phase self-gates: it refuses to touch the GPU unless the 6.10
step-1 validation marker (/tmp/ts_jobs/job610_step1.json) exists with
success=true. If validation failed, the job exits without measuring --
cancel it with `ts -k` (or let it exit 1 here).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GATE_FILE = Path('/tmp/ts_jobs/job610_step1.json')
OUT_DIR = REPO_ROOT / 'runs' / 'stage1_dataset_measured'

OPS = ['scale', 'add', 'saxpy', 'relu', 'mul', 'copy', 'triad', 'div']
GRIDS = [[4, 1, 1], [8, 1, 1], [16, 1, 1], [32, 1, 1]]
BLOCKS = [[128, 1, 1], [256, 1, 1]]
SIZES = [4096, 16384, 65536]


def build_batch(workdir: Path, arch: str = 'sm_120') -> Path:
    """CPU-only: specs -> drivers -> binaries -> manifest. Returns manifest path."""
    from epic_u2o.generator_stage1 import generate_kernels, kernel_hash_for
    from epic_u2o.kernel_driver import synthesize

    workdir = Path(workdir)
    (workdir / 'bin').mkdir(parents=True, exist_ok=True)
    manifest = []
    specs = list(generate_kernels(ops=OPS, grid_sweeps=GRIDS,
                                  block_sweeps=BLOCKS, problem_sizes=SIZES))
    print(f"generated {len(specs)} specs", flush=True)
    jobs = []
    for i, spec in enumerate(specs):
        d = spec.to_dict()
        cu = workdir / f'k{i:04d}.cu'
        exe = workdir / 'bin' / f'k{i:04d}.out'
        cu.write_text(synthesize(d))
        jobs.append((i, d, str(cu), str(exe)))
    # Compile serially here (caller parallelizes by sharding, or run with
    # xargs; kept serial in-process for log clarity).
    for i, d, cu, exe in jobs:
        r = subprocess.run(['nvcc', '-arch=' + arch, '-O2', cu, '-o', exe],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f"COMPILE_FAIL k{i:04d} {d['op_type']}: {r.stderr[:300]}", flush=True)
            continue
        # gpu-sim-9yt: single canonical definition (was a divergent inline
        # recipe missing problem_size).
        khash = kernel_hash_for(d['source'], d['grid'], d['block'],
                                d['problem_size'])
        manifest.append({'idx': i, 'kernel_hash': khash, 'exe': exe,
                         'grid': d['grid'], 'block': d['block'],
                         'shared_mem_bytes': d.get('shared_mem_bytes', 0),
                         'op_type': d['op_type'], 'problem_size': d['problem_size'],
                         'source': d['source']})
    mp = workdir / 'manifest.jsonl'
    with open(mp, 'w') as f:
        for m in manifest:
            f.write(json.dumps({k: v for k, v in m.items() if k != 'source'}) + '\n')
    # Sources archived separately (large).
    with open(workdir / 'sources.jsonl', 'w') as f:
        for m in manifest:
            f.write(json.dumps({'idx': m['idx'], 'kernel_hash': m['kernel_hash'],
                                'source': m['source']}) + '\n')
    print(f"compiled {len(manifest)}/{len(jobs)} -> {mp}", flush=True)
    return mp


def device_snapshot() -> dict:
    """Read-only nvidia-smi snapshot (no GPU workload)."""
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,'
             'clocks.gr,clocks.mem,power.limit', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30)
        return {'nvidia_smi': r.stdout.strip(), 'ts': time.time()}
    except Exception as e:
        return {'nvidia_smi_error': str(e), 'ts': time.time()}


def check_gate() -> None:
    """Refuse to measure unless 6.10 step-1 validation passed."""
    if not GATE_FILE.exists():
        raise SystemExit(f"GATE: {GATE_FILE} missing -- 6.10 step-1 validation "
                         f"has not reported; refusing to measure. Cancel this job.")
    gate = json.loads(GATE_FILE.read_text())
    if not gate.get('success'):
        raise SystemExit(f"GATE: step-1 validation success=false -- refusing to "
                         f"measure. Cancel this job.")
    if gate.get('n_trials', 0) < 10:
        raise SystemExit(f"GATE: step-1 reported only {gate.get('n_trials')} trials -- "
                         f"refusing to measure. Cancel this job.")
    print(f"GATE OK: step-1 success, {gate.get('n_trials')} trials", flush=True)


def run_batch(workdir: Path, timeout_sec: int = 30) -> Path:
    """GPU phase: execute prebuilt binaries, write real measured rows."""
    from epic_u2o.safe_harness import SafeHarness
    from epic_u2o.measurement_stats import compute_trial_stats

    check_gate()
    workdir = Path(workdir)
    manifest = [json.loads(l) for l in open(workdir / 'manifest.jsonl')]
    harness = SafeHarness(timeout_sec=timeout_sec)
    snap = device_snapshot()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'batch.jsonl'
    done = 0
    with open(out_path, 'a') as out:
        for m in manifest:
            res = harness.run_kernel(m['exe'])
            if not res.success:
                print(f"RUN_FAIL {m['kernel_hash']} {m['op_type']}: "
                      f"rc={res.returncode} timeout={res.timeout} "
                      f"{res.stderr[:120]}", flush=True)
                continue
            trials = []
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.startswith('{'):
                    try:
                        trials.append(float(json.loads(line)['time_ms']))
                    except (ValueError, KeyError):
                        pass
            if len(trials) < 10:
                print(f"FEW_TRIALS {m['kernel_hash']}: {len(trials)}", flush=True)
                continue
            st = compute_trial_stats(trials)
            row = {'kernel_hash': m['kernel_hash'], 'op_type': m['op_type'],
                   'launch_config': {'grid': m['grid'], 'block': m['block'],
                                     'shared_mem_bytes': m['shared_mem_bytes']},
                   'problem_size': m['problem_size'],
                   'median_runtime_ms': st.median_ms, 'iqr_ms': st.iqr_ms,
                   'n_trials': st.n_trials, 'cv': st.cv,
                   'reliable': st.is_reliable,
                   'device_prop_snapshot': snap}
            out.write(json.dumps(row) + '\n')
            out.flush()
            done += 1
            if done % 25 == 0:
                print(f"measured {done}/{len(manifest)}", flush=True)
    print(f"BATCH DONE: {done}/{len(manifest)} rows -> {out_path}", flush=True)
    return out_path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='First real batch: build (CPU) or run (GPU via ts)')
    p.add_argument('phase', choices=['build', 'run'])
    p.add_argument('--workdir', default='/tmp/ts_batch1')
    p.add_argument('--timeout', type=int, default=30)
    a = p.parse_args()
    if a.phase == 'build':
        build_batch(Path(a.workdir))
    else:
        run_batch(Path(a.workdir), timeout_sec=a.timeout)
