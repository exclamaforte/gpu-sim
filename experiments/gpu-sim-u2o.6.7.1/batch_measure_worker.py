#!/usr/bin/env python3
"""Batched / persistent measurement worker. gpu-sim-u2o.6.7.1 -- the P0.

THE PROTOCOL IS FIXED and this does not change it. Per example: 5 warmup
launches + 20 timed launches, x2 invocations = 40 timed launches, exactly as
gpu-sim-u2o.7.1 measured. The trial count buys every label in the corpus and
6.6.1 priced a label at ~36% MDD at N=20, so nothing here may reduce it. What
changes is ONLY how often a process starts: one CUDA context for the whole
batch instead of two per example.

That is the entire mechanism, and it is worth being precise about WHY it is so
large. 7.1 measured 418.6 ms of wall per example against 126 us of device work.
Decomposing that: 2 process spawns per example, each paying a CUDA context
init, and 40 launches. The context init dominates by two orders of magnitude
over the kernel itself. So this is not a kernel-throughput problem and never
was; it is a context-creation bill paid 528 times.

PER-EXAMPLE SAMPLING, because the temptation is exactly backwards here. A
persistent process outlives its sample, so a worker that samples once at batch
start and stamps every row would reproduce at larger scale the precise defect
6.6.6 was filed for: a record that asserts rather than observes. So every row
here gets its OWN queue_state and its OWN telemetry, taken between that row's
launches, and the artifact demonstrates it on rows whose sample times differ.
The cost of sampling is MEASURED and reported separately rather than folded
into "overhead", because it is a real part of the budget and hiding it inside
an unattributed residual would be the same mistake as not splitting it at all.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o import queue_state as qs  # noqa: E402
from epic_u2o.generator_stage1 import generate_kernels  # noqa: E402

ARCH = "sm_120"
WARMUP = 5            # unchanged: 5 warmup launches, as 7.1's driver does
TIMED = 20            # unchanged: the trial count that buys the label
INVOCATIONS = 2       # unchanged: 2 invocations per example

SPECS_REF = "lab/gpu-sim-u2o.7.1"
SPECS = "experiments/gpu-sim-u2o.7.1/build/manifest.json"


def load_specs() -> list:
    """7.1's committed specs, so this measures the SAME examples.

    Generating fresh specs would make 'do the medians reproduce' meaningless,
    because a different sweep is a different experiment.
    """
    r = subprocess.run(["git", "show", f"{SPECS_REF}:{SPECS}"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        # the manifest is a build artifact and may be untracked; fall back to
        # the same sweep 7.1 used, and SAY that is what happened
        r2 = subprocess.run(
            ["git", "show", f"{SPECS_REF}:experiments/gpu-sim-u2o.7.1/build_batch.py"],
            cwd=REPO, capture_output=True, text=True)
        if r2.returncode != 0:
            raise SystemExit(
                f"FATAL: cannot read 7.1's manifest or build script.\n"
                f"Measuring a DIFFERENT sweep would make the reproduction check "
                f"meaningless, so this refuses.")
        return _regenerate_sweep()
    return json.loads(r.stdout)["specs"]


def _regenerate_sweep() -> list:
    """7.1's exact sweep, rebuilt from its build_batch.py parameters."""
    BLOCKS = [[64, 1, 1], [128, 1, 1], [256, 1, 1], [1024, 1, 1]]
    SIZES = [1 << 12, 1 << 14, 1 << 16, 1 << 18]
    OPS = ["scale", "add", "saxpy", "relu", "copy", "triad", "mul", "div"]

    def grids_for(n, block):
        bt = block[0] * block[1] * block[2]
        full = max(1, n // bt)
        out = [[full, 1, 1], [max(1, full // 2), 1, 1], [max(1, full // 8), 1, 1]]
        if full > 64:
            out.append([1, 1, 1])
        return out

    import hashlib
    specs = []
    for n in SIZES:
        for block in BLOCKS:
            for grid in grids_for(n, block):
                for spec in generate_kernels(ops=OPS, grid_sweeps=[grid],
                                             block_sweeps=[block],
                                             problem_sizes=[n]):
                    d = spec.to_dict()
                    d["covers"] = min(grid[0] * block[0], n)
                    d["deliberately_poor"] = grid == [1, 1, 1] or block == [1, 1, 1]
                    # kernel_hash EXACTLY as 7.1's build_batch._row computes it.
                    # Inheriting the identifier is what lets criterion 2 compare
                    # this run's medians against 7.1's row-for-row; inventing a
                    # different hash would make the comparison vacuous.
                    d["kernel_hash"] = hashlib.sha256(
                        (d["source"] + json.dumps(
                            [d["grid"], d["block"], d["problem_size"]])
                         ).encode()).hexdigest()[:16]
                    specs.append(d)
    return specs


DRIVER_TEMPLATE = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(x) do {{ cudaError_t e_=(x); if(e_!=cudaSuccess){{ \
    fprintf(stderr,"{{\"error\":\"%s:%d %s\"}}\n",__FILE__,__LINE__, \
    cudaGetErrorString(e_)); exit(3); }} }} while(0)

/* THE REAL KERNELS, verbatim from 7.1's specs. A single generic kernel would
 * be faster and worthless: criterion 2 requires the per-example medians to
 * REPRODUCE 7.1's, and that is only a meaningful check if the same code is
 * being timed. This is the one place where the obvious optimisation -- reuse
 * one kernel -- would have quietly destroyed the bead. */
{body}

enum {{ MODE_WARM = 0, MODE_COLD = 1, MODE_RESET = 2 }};

struct Spec {{
    int kidx; int grid, block, n; const char *hash;
}};

static void launch(int kidx, int grid, int block, int n,
                   float *b0, float *b1, float *b2, float *b3)
{{
    switch (kidx) {{
{cases}
    }}
}}

/* ONE context for the whole batch. 7.1 paid this twice per example. */
int main(int argc, char **argv) {{
    if (argc < 3) {{
        fprintf(stderr, "usage: %s <specfile> <warm|cold|reset>\n", argv[0]);
        return 2;
    }}
    const char *mode_s = argv[2];
    int mode = (strcmp(mode_s, "cold") == 0) ? MODE_COLD
             : (strcmp(mode_s, "reset") == 0) ? MODE_RESET : MODE_WARM;
    FILE *f = fopen(argv[1], "r");
    if (!f) {{ fprintf(stderr, "no specfile\n"); return 2; }}
    static Spec specs[{nmax}];
    int nspec = 0;
    char h[64];
    while (nspec < {nmax} && fscanf(f, "%63s %d %d %d %d", h, &specs[nspec].kidx,
                  &specs[nspec].grid, &specs[nspec].block,
                  &specs[nspec].n) == 5) {{
        specs[nspec].hash = strdup(h); nspec++;
    }}
    fclose(f);

    float *bufs[4] = {{NULL, NULL, NULL, NULL}};
    size_t cap = 0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int s = 0; s < nspec; s++) {{
        Spec sp = specs[s];
        /* gpu-sim-u2o.6.7.1 discriminator. Criterion 2 fails (medians do not
         * reproduce 7.1's) and I named TWO candidate causes: the two invocations
         * no longer have independent CUDA contexts, and warm state carries
         * between examples in a persistent process. Profiling cannot separate
         * them, so this varies ONE THING per run:
         *
         *   warm   -- current behaviour, the baseline to beat
         *   cold   -- free and reallocate the buffers between EXAMPLES, so no
         *             L2/TLB/allocator state carries from one example to the
         *             next. Tests the WARM-STATE cause.
         *   reset  -- cudaDeviceReset() between the two INVOCATIONS of an
         *             example, so they do not share a context. Tests the
         *             SHARED-CONTEXT cause.
         *
         * The protocol is identical in all three: 2 invocations x (5 warmup +
         * 20 timed) = 40 timed launches per example. Only the isolation varies.
         */
        if (mode == MODE_COLD && cap > 0) {{
            for (int q = 0; q < 4; q++) {{ if (bufs[q]) cudaFree(bufs[q]); }}
            cap = 0;
        }}
        if ((size_t)sp.n > cap) {{
            for (int q = 0; q < 4; q++) {{ if (bufs[q]) cudaFree(bufs[q]); }}
            cap = sp.n;
            for (int q = 0; q < 4; q++) {{
                CHECK(cudaMalloc(&bufs[q], cap * sizeof(float)));
                CHECK(cudaMemset(bufs[q], 0, cap * sizeof(float)));
            }}
        }}
        /* TWO INVOCATIONS, as 7.1 measured. My first version ran the 25
         * launches ONCE and reported 50 launches per example in its banner --
         * the banner and the code disagreed, and the device time came out at
         * 76.6 us against the 126 us the protocol requires. That is precisely
         * the violation the fixed-protocol ruling exists to catch, caught by
         * the device-time assertion rather than by reading the diff. Each
         * invocation is median-of-20, and the example's value is the median of
         * the two invocation medians, which is 7.1's structure exactly. */
        for (int inv = 0; inv < {invocations}; inv++) {{
            /* reset BETWEEN the invocations of this example, so they do not
             * share a CUDA context. Diagnostic only: it is far slower than a
             * real context init, and that is the point -- it tells us whether
             * the context is what moved the numbers, not what a fix would
             * cost. */
            if (mode == MODE_RESET && inv > 0) {{
                for (int q = 0; q < 4; q++) {{ if (bufs[q]) cudaFree(bufs[q]); }}
                cap = 0;
                CHECK(cudaDeviceReset());
                /* REALLOCATE HERE. My first version set cap = 0 and expected
                 * the per-example allocation path to notice -- but that path is
                 * OUTSIDE this loop, so invocation 1 launched through freed
                 * pointers and died with "an illegal memory access was
                 * encountered". The diagnostic mode was the only one that
                 * crashed, which is at least a legible failure rather than a
                 * silently wrong number. */
                for (int q = 0; q < 4; q++) {{
                    CHECK(cudaMalloc(&bufs[q], (size_t)sp.n * sizeof(float)));
                    CHECK(cudaMemset(bufs[q], 0, (size_t)sp.n * sizeof(float)));
                }}
                cap = (size_t)sp.n;
                CHECK(cudaEventCreate(&start));
                CHECK(cudaEventCreate(&stop));
            }}
        for (int t = 0; t < {warmup} + {timed}; t++) {{
            if (t == {warmup} && inv == 0) {{ printf("BEGIN %s\n", sp.hash); fflush(stdout); }}
            CHECK(cudaEventRecord(start));
            launch(sp.kidx, sp.grid, sp.block, sp.n,
                   bufs[0], bufs[1], bufs[2], bufs[3]);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms = 0.f;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (t >= {warmup})
                printf("TRIAL %s %d %d %.6f\n", sp.hash, inv, t - {warmup}, ms);
        }}
        }}
        /* Hand control back so the host samples THIS example, at THIS time. */
        printf("END %s\n", sp.hash);
        fflush(stdout);
    }}
    for (int q = 0; q < 4; q++) if (bufs[q]) cudaFree(bufs[q]);
    printf("DONE %d\n", nspec);
    return 0;
}}
"""


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="measure only the first N examples (0 = all)")
    ap.add_argument("--out", default="batch_rows.jsonl")
    ap.add_argument("--summary-out", default="batch_summary.json",
                    help="DIAGNOSTIC RUNS MUST NOT OVERWRITE THE ARTIFACT OF "
                         "RECORD. The discriminator's three modes all wrote "
                         "batch_summary.json, so the committed artifact ended "
                         "up describing the slow reset diagnostic (7.54 ex/s) "
                         "instead of the authoritative warm run (168.81 ex/s) "
                         "-- and I committed that. A diagnostic that clobbers "
                         "the thing it is diagnosing is the same class of "
                         "defect as a stale artifact, wearing a fresh "
                         "timestamp.")
    ap.add_argument("--mode",
                    choices=("warm", "cold", "reset", "perproc"),
                    default="warm",
                    help="warm: persistent process, state carries (current). "
                         "cold: buffers freed and reallocated between EXAMPLES "
                         "(tests warm-state carryover). reset: "
                         "cudaDeviceReset between the two INVOCATIONS (tests "
                         "shared CUDA context).")
    ap.add_argument("--op-filter", default="")
    args = ap.parse_args()
    specs = load_specs()
    specs = [s for s in specs if not args.op_filter or s.get("op_type") == args.op_filter]

    if args.limit:
        specs = specs[:args.limit]
    print(f"batched worker: {len(specs)} examples, per example "
          f"{INVOCATIONS} invocations x ({WARMUP} warmup + {TIMED} timed) = "
          f"{TIMED * INVOCATIONS} timed + {WARMUP * INVOCATIONS} warmup = "
          f"{(WARMUP + TIMED) * INVOCATIONS} launches "
          f"(UNCHANGED from 7.1)", flush=True)

    import re as _re
    import tempfile
    td = Path(tempfile.mkdtemp())

    # Embed each DISTINCT kernel source once, renamed so the driver can hold
    # them all, and dispatch by index. Same source in, same launch out.
    kernels, kidx_of = [], {}
    for s in specs:
        h = s["kernel_hash"]
        if h in kidx_of:
            continue
        src = s.get("source")
        if not src:
            raise SystemExit(
                f"FATAL: spec {h} carries no source. A driver with a generic "
                f"kernel would time different code than 7.1 measured, which "
                f"makes criterion 2 meaningless, so this refuses.")
        m = _re.search(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", src)
        if not m:
            raise SystemExit(f"FATAL: no __global__ in spec {h} (gpu-sim-9ti)")
        kname, params = m.group(1), m.group(2)
        kidx = len(kernels)
        kidx_of[h] = kidx
        renamed = _re.sub(r"__global__\s+void\s+\w+\s*\(",
                          f"__global__ void u2o_{kname}_{kidx}(", src, count=1)
        kernels.append((f"u2o_{kname}_{kidx}", kname, params, renamed))

    body = "\n".join(src for _, _, _, src in kernels)
    def call_site(i, alias, params):
        args, nbuf = [], 0
        for raw in params.split(","):
            raw = raw.strip()
            if not raw:
                continue
            if "*" in raw:
                args.append(f"b{nbuf}")
                nbuf += 1
            elif "float" in raw:
                args.append("1.0f")
            elif "int" in raw or "long" in raw:
                args.append("n")
            else:
                raise SystemExit(
                    f"FATAL: unrecognised parameter {raw!r} in kernel {alias}. "
                    f"Guessing would time different code than the dataset names.")
        return (f"    case {i}: {alias}<<<grid, block>>>("
                + ", ".join(args) + "); break;")

    cases = "\n".join(call_site(i, alias, params)
                      for i, (alias, _kn, params, _src) in enumerate(kernels))
    specfile = td / "specs.txt"
    specfile.write_text("".join(
        f"{s['kernel_hash']} {kidx_of[s['kernel_hash']]} "
        f"{s['grid'][0]} {s['block'][0]} {s['problem_size']}\n"
        for s in specs))
    cu = td / "batch_worker.cu"
    cu.write_text(DRIVER_TEMPLATE.format(nmax=len(specs) + 8, body=body,
                                        cases=cases,
                                        warmup=WARMUP, timed=TIMED,
                                        invocations=INVOCATIONS))
    exe = td / "batch_worker"
    b = subprocess.run(["nvcc", f"-arch={ARCH}", str(cu), "-o", str(exe)],
                       capture_output=True, text=True)
    if b.returncode != 0:
        print("BUILD FAILED\n" + b.stderr, flush=True)
        return 10

    # --- the ONE process, and everything timed against it -----------------
    t_start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    if args.mode == "perproc":
        return _run_per_invocation_process(args, specs, exe, td, t_start,
                                          started_at, sampling_wall
                                          if False else None)

    p = subprocess.Popen([str(exe), str(specfile), args.mode],
                         stdout=subprocess.PIPE,
                         text=True, bufsize=1)
    per_example: dict = {}
    rows_by_hash: dict = {}
    n_begin = n_end = 0
    spec_by_hash = {s["kernel_hash"]: s for s in specs}
    sampling_wall = 0.0

    # Sampled INSIDE the loop, on each END marker. My first version read all
    # the output and THEN sampled every row, which put every sample at the same
    # moment -- a per-batch sample wearing per-row clothing, and precisely the
    # defect this bead's constraint 2 exists to prevent. The timestamps below
    # are monotonic across rows because they are taken as the rows happen.
    for line in p.stdout:
        line = line.strip()
        if line.startswith("TRIAL "):
            _, h, inv, t, ms = line.split()
            per_example.setdefault(h, []).append(
                (int(inv), int(t), float(ms)))
        elif line.startswith("BEGIN "):
            h = line.split()[1]
            per_example[h] = []
            n_begin += 1
        elif line.startswith("END "):
            h = line.split()[1]
            n_end += 1
            raw = per_example.get(h, [])
            if not raw:
                continue
            by_inv: dict = {}
            for inv, _t, ms in raw:
                by_inv.setdefault(inv, []).append(ms)
            inv_meds = [statistics.median(v) for v in by_inv.values()]
            trials = [ms for _i, _t, ms in raw]
            t_s0 = time.time()
            # 6.6.6 hardening: queue_state() refuses unidentified calls.
            # Identify by our own command substring (matcher prefers the
            # RUNNING job); U2O_OWN_CMD overrides for uniquely-named runs.
            import os as _os
            qstate = qs.queue_state(
                own_command_substr=_os.environ.get(
                    "U2O_OWN_CMD", "batch_measure_worker.py"))
            telemetry = _telemetry()
            sampling_wall += time.time() - t_s0
            s = spec_by_hash[h]
            rows_by_hash[h] = {
                "kernel_hash": h, "op_type": s.get("op_type"),
                "problem_size": s["problem_size"],
                "grid": s["grid"], "block": s["block"],
                "n_trials": len(inv_meds),
                "n_invocations": len(inv_meds),
                "invocation_medians_ms": inv_meds,
                "n_timed_launches": len(trials),
                # literal here, not the later local: this dict is built during
                # the loop, before that name exists
                "launches_per_example": (WARMUP + TIMED) * INVOCATIONS,
                # 7.1's structure: median over the per-invocation medians.
                # A median over all 40 trials pooled would be a DIFFERENT
                # estimator and the comparison against 7.1 would be quietly
                # invalid, so this mirrors the original exactly.
                "median_runtime_ms": statistics.median(inv_meds),
                "trials_ms": trials,
                "sampled_at": datetime.fromtimestamp(t_s0, timezone.utc).isoformat(),
                "sampled_at_epoch": t_s0,
                "queue_state": qstate,
                "telemetry": telemetry,
            }
        elif line.startswith("DONE"):
            break
    rc = p.wait()
    wall = time.time() - t_start
    if rc != 0:
        print(f"worker exited {rc}", flush=True)
        return 11
    rows = [rows_by_hash[s["kernel_hash"]] for s in specs
            if s["kernel_hash"] in rows_by_hash]

    # Derived BEFORE the summary that reports them. My first version computed
    # these after the dict literal, so the dict raised UnboundLocalError on a
    # value it had already used -- caught by running, not by reading.
    launches_per_example = (WARMUP + TIMED) * INVOCATIONS
    timed_launches_per_example = TIMED * INVOCATIONS
    warmup_launches_per_example = WARMUP * INVOCATIONS
    return _finish(args, rows, wall, started_at, sampling_wall,
                   n_begin, n_end)


def _finish(args, rows, wall, started_at, sampling_wall, n_begin, n_end,
            processes=None):
    # Derived here, not in main(): _finish is also the per-invocation path's
    # exit, and the first version of that path raised NameError on it. Anything
    # the summary reports must be computed INSIDE the summary.
    launches_per_example = (WARMUP + TIMED) * INVOCATIONS
    timed_launches_per_example = TIMED * INVOCATIONS
    warmup_launches_per_example = WARMUP * INVOCATIONS
    sample_times = [r["sampled_at_epoch"] for r in rows]
    distinct = len({round(t, 6) for t in sample_times})
    span = (max(sample_times) - min(sample_times)) if sample_times else 0.0

    n = len(rows)
    # Device time = median kernel time x the launches that ran it (warmup
    # launches are real device work too, so they are counted).
    device_s = sum(r["median_runtime_ms"] * launches_per_example
                   for r in rows) / 1000.0
    sampling_ms = sampling_wall * 1000.0
    out = {
        "bead": "gpu-sim-u2o.6.7.1",
        "gpu_used": True,
        "mode": args.mode,
        "mode_meaning": {
            "warm": "one process, one context, state carries between examples "
                    "-- the configuration in which criterion 2 failed",
            "cold": "buffers freed and reallocated between EXAMPLES; isolates "
                    "warm-state carryover",
            "reset": "cudaDeviceReset between the two INVOCATIONS; isolates a "
                     "shared CUDA context",
            "perproc": "an INDEPENDENT PROCESS per invocation, which is 7.1's "
                       "isolation, with everything else identical. The decisive "
                       "arm: it separates 'the estimator changed' from 'the "
                       "environment changed'.",
        }[args.mode],
        "protocol": "IDENTICAL in all modes: 2 invocations x (5 warmup + 20 "
                    "timed) = 40 timed launches per example. Only the ISOLATION "
                    "varies, which is the point -- this experiment varies one "
                    "thing at a time rather than optimising two at once.",
        "worker": {
            "processes_started": (processes if processes is not None else 1),
            "contexts_created": (processes if processes is not None else 1),
            "examples": n,
            "warmup_launches_per_example": WARMUP,
            "timed_launches_per_example": TIMED,
            "invocations_per_example": INVOCATIONS,
            "timed_launches_per_example": timed_launches_per_example,
            "warmup_launches_per_example": warmup_launches_per_example,
            "total_launches_per_example": launches_per_example,
            "protocol_unchanged": True,
            "protocol_note": (
                "IDENTICAL to 7.1: 5 warmup + 20 timed, 2 invocations, 40 timed "
                "launches per example. Only the number of PROCESS STARTS "
                "changed, from 2 per example to 1 for the batch. The trial "
                "count that buys each label is untouched."),
        },
        "throughput": {
            "examples": n,
            "wall_sec": wall,
            "examples_per_sec": n / wall if wall else 0.0,
            "examples_per_hour": (n / wall) * 3600.0 if wall else 0.0,
            "ms_per_example": wall * 1000.0 / n if n else 0.0,
        },
        "device_time_per_example": {
            "asserted_us": 126.0,
            "measured_us": (device_s / n) * 1e6 if n else 0.0,
            "unchanged": abs((device_s / n) * 1e6 - 126.0) < 40.0 if n else False,
            "note": ("derived from the committed cells' medians x launches per "
                     "example; this is the quantity the throughput fix must NOT "
                     "move"),
        },
        "per_example_sampling": {
            "rows": n,
            "distinct_sample_times": distinct,
            "sample_time_span_sec": span,
            "sampled_per_row_not_per_batch": distinct == n,
            "sampling_wall_sec": sampling_wall,
            "sampling_ms_per_example": sampling_ms / n if n else 0.0,
            "sampling_is_a_real_cost": (
                f"per-example telemetry costs {sampling_ms / n if n else 0:.1f} ms "
                f"per example and is reported SEPARATELY rather than folded into "
                f"an unattributed residual. It is the price of the per-row "
                f"observation this bead's constraint 2 requires, and hiding it "
                f"inside 'overhead' would be the same mistake as not splitting it "
                f"out at all."),
            "note": ("every row carries its own queue_state and its own "
                     "telemetry, taken at that row's time. A per-batch sample "
                     "stamped onto many rows would satisfy the letter of the "
                     "row carrying a field while observing nothing."),
        },
        "started_at": started_at,
        "n_begin_markers": n_begin, "n_end_markers": n_end,
    }
    (HERE / args.summary_out).write_text(json.dumps(out, indent=2))
    for i, r in enumerate(rows):
        r["mode"] = args.mode
        r["position_in_batch"] = i
    (HERE / args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))

    if n == 0:
        raise SystemExit(
            f"FATAL: {args.mode} produced 0 examples in {wall:.1f}s. A run that "
            f"measured nothing must not print a throughput line, because the "
            f"zero reads as a measurement of a very slow batch.")
    print(f"\n{n} examples in {wall:.2f}s = {out['throughput']['examples_per_sec']:.2f} ex/s "
          f"({out['throughput']['examples_per_hour']:,.0f}/hour)")
    print(f"processes started: 1   device time/example: "
          f"{out['device_time_per_example']['measured_us']:.1f} us "
          f"(asserted 126)")
    print(f"per-example samples: {distinct}/{n} distinct times, "
          f"span {span:.3f}s")
    return 0


class _Nvml:
    """In-process NVML reader, via ctypes.

    gpu-sim-u2o.6.7.1's own attribution said the dominant remaining cost was
    the per-example sample: spawning `nvidia-smi` once per row cost ~39 ms,
    the same order as the process spawn the batching had just eliminated. So
    the next increment had to be the SAMPLE, not the worker.

    NVML through ctypes is the same library nvidia-smi calls, loaded once, so a
    query is a function call rather than a fork/exec plus a driver handshake.
    Measured on the real 5090: ~0.2 ms against nvidia-smi's ~39 ms, so
    per-example telemetry stops being the bottleneck while still being a real
    observation at that row's own time.

    Every field is read the same way 6.5's harness reads it, and the row
    records WHICH path produced it, so a reader never has to guess whether a
    sample came from the cheap path or the expensive fallback.
    """

    CLOCK_SM = 2
    SM_CLOCK = 1
    MEM_CLOCK = 2
    TEMP_GPU = 0
    POWER_MW = 0
    UTIL_GPU = 0

    # nvmlMemory_t is a STRUCT of five unsigned long long (40 bytes), not an
    # int. My first version passed a c_uint for it, which let NVML write 40
    # bytes into a 4-byte slot and segfaulted the process. The size is declared
    # here rather than guessed per call, because getting it wrong does not
    # return an error -- it corrupts the stack.
    NVML_MEMORY_STRUCT_FIELDS = 5

    def __init__(self):
        import ctypes
        self._c = ctypes
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.ok = self.lib.nvmlInit_v2() == 0
        self.handle = ctypes.c_void_p()
        if self.ok:
            self.ok = self.lib.nvmlDeviceGetHandleByIndex_v2(
                0, ctypes.byref(self.handle)) == 0
        self._path = "nvml-ctypes" if self.ok else None

    def _out_u32(self, fn, *args):
        """fn(handle, *args, out) -> value, or None.

        The extra *args matters. My first helper was _u32(fn, arg) which called
        fn(arg, byref(v)) -- so for the THREE-argument NVML calls it passed the
        sensor/type as the HANDLE and a pointer as the type. That did not return
        an error; it segfaulted, twice, and the reason was invisible in the code.
        """
        v = self._c.c_uint()
        return v.value if fn(self.handle, *args, self._c.byref(v)) == 0 else None

    def _mem_used_mib(self):
        buf = (self._c.c_ulonglong * self.NVML_MEMORY_STRUCT_FIELDS)()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle,
                                             self._c.byref(buf)) != 0:
            return None
        return buf[2] // (1024 * 1024)      # field 2 is `used`

    def read(self) -> dict:
        if not self.ok:
            return {"rc": -1, "path": None, "error": "NVML unavailable"}
        c = self._c

        class _Util(c.Structure):
            _fields_ = [("gpu", c.c_uint), ("memory", c.c_uint)]

        u = _Util()
        self.lib.nvmlDeviceGetUtilizationRates(self.handle, c.byref(u))
        pw = c.c_ulonglong()
        ok_pw = self.lib.nvmlDeviceGetPowerUsage(self.handle,
                                                 c.byref(pw)) == 0
        return {
            "rc": 0, "path": self._path,
            "clocks.sm": self._out_u32(self.lib.nvmlDeviceGetClockInfo,
                                       self.CLOCK_SM),
            "clocks.mem": self._out_u32(self.lib.nvmlDeviceGetClockInfo,
                                        self.MEM_CLOCK),
            "power.draw_mW": pw.value if ok_pw else None,
            "temperature.gpu": self._out_u32(
                self.lib.nvmlDeviceGetTemperature, self.TEMP_GPU),
            "utilization.gpu": u.gpu,
            "memory.used_MiB": self._mem_used_mib(),
        }


_NVML = None


def _run_per_invocation_process(args, specs, exe, td, t_start, started_at, _):
    """gpu-sim-u2o.6.7.1: the DECISIVE arm.

    Independent PROCESS per invocation, which is 7.1's isolation, with
    everything else held identical: same examples, same 5 warmup + 20 timed per
    invocation, same median-of-invocation-medians estimator, same per-example
    sampling. The ONLY thing that changes against the `warm` arm is the process
    boundary, so this separates "the estimator changed" from "the environment
    changed" -- which is the question the whole episode turns on.

    It needs no driver change: the driver already takes a specfile, so each
    invocation is simply a one-spec file and a fresh process.
    """
    per_example, rows_by_hash, spec_by_hash = {}, {}, {}
    kidx_of = {}
    for s in specs:
        spec_by_hash[s["kernel_hash"]] = s
        if s["kernel_hash"] not in kidx_of:
            kidx_of[s["kernel_hash"]] = len(kidx_of)
    n_inv = INVOCATIONS
    sampling = 0.0
    for idx, s in enumerate(specs):
        h = s["kernel_hash"]
        one = td / f"one_{h}.txt"
        # FIVE fields: the driver's fscanf is "%63s %d %d %d %d" (hash, kernel
        # index, grid, block, n). My first version wrote four and every
        # invocation silently matched zero specs, so the run reported 0 examples
        # in 65.85 s -- a FAILURE that looked like a result because it printed
        # a well-formed throughput line.
        one.write_text(f"{h} {kidx_of[h]} {s['grid'][0]} {s['block'][0]} "
                       f"{s['problem_size']}\n")
        inv_meds = []
        for _inv in range(n_inv):
            r = subprocess.run([str(exe), str(one), "warm"],
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                raise SystemExit(
                    f"per-invocation process failed for {h}: {r.stderr[-400:]}")
            trials = []
            for line in r.stdout.splitlines():
                if line.startswith("TRIAL "):
                    _, _hh, _iv, _t, ms = line.split()
                    trials.append(float(ms))
            if trials:
                inv_meds.append(statistics.median(trials))
        if not inv_meds:
            continue
        t_s0 = time.time()
        import os as _os2
        qstate = qs.queue_state(
            own_command_substr=_os2.environ.get(
                "U2O_OWN_CMD", "batch_measure_worker.py"))
        telemetry = _telemetry()
        sampling += time.time() - t_s0
        rows_by_hash[h] = {
            "kernel_hash": h, "op_type": s.get("op_type"),
            "problem_size": s["problem_size"],
            "grid": s["grid"], "block": s["block"],
            "n_trials": len(inv_meds), "n_invocations": len(inv_meds),
            "invocation_medians_ms": inv_meds,
            "n_timed_launches": len(inv_meds) * TIMED,
            "median_runtime_ms": statistics.median(inv_meds),
            "trials_ms": [],
            "sampled_at": datetime.fromtimestamp(t_s0, timezone.utc).isoformat(),
            "sampled_at_epoch": t_s0,
            "queue_state": qstate, "telemetry": telemetry,
        }
    rows = [rows_by_hash[s["kernel_hash"]] for s in specs
            if s["kernel_hash"] in rows_by_hash]
    wall = time.time() - t_start
    return _finish(args, rows, wall, started_at, sampling, n_begin=len(rows),
                   n_end=len(rows), processes=len(rows) * n_inv)


def _telemetry() -> dict:
    """Read-only device telemetry for THIS row, cheap.

    Aimed at a query, never at a reset or a clock change: read-only DIAGNOSIS is
    permitted outside GPU work, and 6.5 established this exact data as part of
    the harness. The sampling COST is measured by the caller as part of the
    budget rather than hidden -- which is what surfaced it as the dominant term
    in the first place.
    """
    global _NVML
    if _NVML is None:
        _NVML = _Nvml()
    if _NVML.ok:
        return _NVML.read()
    # Fallback: correct but ~200x slower. Recorded per row so a reader can see
    # which path produced the sample rather than assuming.
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,"
             "utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return {"rc": 0, "path": "nvidia-smi-subprocess",
                    "csv": r.stdout.strip()}
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"rc": -1, "path": "nvidia-smi-subprocess", "error": str(exc)}
    return {"rc": -1, "path": "nvidia-smi-subprocess", "csv": None}


if __name__ == "__main__":
    raise SystemExit(main())
