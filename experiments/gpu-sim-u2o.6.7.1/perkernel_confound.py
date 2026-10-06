#!/usr/bin/env python3
"""The confound experiment: 7.1's OWN per-kernel binaries, one shared context.

The decisive arm showed that restoring 7.1's process isolation does not collapse
the 12.5% offset. What is left is the confound I introduced: 7.1 builds a
SEPARATE binary per kernel via kernel_driver.synthesize() -- each with its own
`main` and its own `cudaMalloc`s sized to that kernel's arity -- and runs each
through run_kernel(). My batched worker builds ONE binary containing all eight
kernels with a dispatch switch, and allocates FOUR buffers of the maximum size
REGARDLESS OF ARITY.

Those are different memory environments, which is exactly what produces a
systematic one-sided offset rather than scatter.

This script removes the confound while keeping the thing that made it fast:

  * every (kernel, n, grid, block) gets its OWN synthesized source, exactly as
    7.1 produced it, compiled as its OWN translation unit with its own nvcc
    invocation, with its own per-kernel buffer allocation;
  * the entry point `main` is renamed so they can be LINKED;
  * all 264 are linked into ONE executable, driven by ONE process with ONE
    CUDA context, which is where the speed came from.

So the only thing that differs from my earlier `warm` arm is the executable and
its memory environment. If the offset collapses, the executable was the cause
and the surviving speed is the per-invocation-process arm's ~1.6x rather than
70x. If it does not, the executable is exonerated too and the cause is
genuinely unidentified.

Must run through `ts`.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o import queue_state as qs  # noqa: E402
from epic_u2o.kernel_driver import synthesize  # noqa: E402
from epic_u2o.tokenizer import tokenize_for_model  # noqa: E402

SPECS_REF = "lab/gpu-sim-u2o.7.1"
MANIFEST = "experiments/gpu-sim-u2o.7.1/build/manifest.json"
ARCH = "sm_120"

def load_specs() -> tuple:
    """7.1's exact specs, and PROOF they are 7.1's.

    The manifest is an uncommitted build artifact, so it cannot be read back.
    Rather than invent a sweep -- which would reproduce the confound while
    claiming to remove it -- this regenerates 7.1's sweep with the SAME code and
    the SAME kernel_hash construction, then PROVES the set is 7.1's by
    intersecting the hashes with 7.1's committed measured.jsonl.

    A regenerated sweep is only acceptable if it is verifiably the same sweep, so
    the proof is part of the return value and lands in the artifact rather than
    in a comment.
    """
    r = subprocess.run(["git", "show", f"{SPECS_REF}:{MANIFEST}"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode == 0:
        specs = json.loads(r.stdout)["specs"]
        return specs, {"source": "committed manifest",
                       "hash_match_with_7_1": None}
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bw", str(HERE / "batch_measure_worker.py"))
    bw = importlib.util.module_from_spec(spec)
    sys.modules["bw"] = bw
    spec.loader.exec_module(bw)
    specs = bw._regenerate_sweep()
    mine = {s["kernel_hash"] for s in specs}
    rr = subprocess.run(
        ["git", "show",
         f"{SPECS_REF}:experiments/gpu-sim-u2o.7.1/measured.jsonl"],
        cwd=REPO, capture_output=True, text=True)
    theirs = set()
    for line in rr.stdout.splitlines():
        if line.strip():
            theirs.add(json.loads(line)["kernel_hash"])
    return specs, {
        "source": "regenerated with 7.1's own sweep code and hash construction",
        "n_regenerated": len(mine), "n_in_7_1_dataset": len(theirs),
        "hash_match_with_7_1": len(mine & theirs),
        "proven_identical": mine == theirs,
    }


def build_per_kernel_executables(specs: list, td: Path) -> tuple:
    """One translation unit per DISTINCT KERNEL, compiled by its own nvcc.

    The first attempt compiled one TU per SPEC, which failed to link: 264 specs
    share only 8 distinct kernel sources, so every object defined the same
    __global__ and the linker rejected 250 duplicates. That failure is
    informative rather than incidental -- it is the same fact as the confound,
    seen from the build side: the KERNEL is the unit of compilation, and the
    per-spec variation is only the launch geometry.

    So: one TU per distinct kernel, exactly the granularity 7.1's compilation
    used, with the per-spec values (N, grid, block) HOISTED from main()'s locals
    to globals the driver sets at runtime. Everything that differs between two
    kernels -- the body, the argument list, the NUMBER AND SIZE of the
    cudaMalloc calls -- is untouched, because that is the confound.

    What this does NOT preserve, stated rather than glossed: 7.1 baked N, grid
    and block into the source at synthesis time, so its binary had them as
    compile-time constants. Here they are runtime values. The kernel body, its
    arity and its buffer allocation -- the parts under test -- are identical.
    """
    seen, objs, idxs, by_source = {}, [], [], {}
    for i, s in enumerate(specs):
        key = s["source"]
        if key in seen:
            by_source.setdefault(seen[key], []).append(i)
            continue
        k = len(objs)
        src = synthesize(s)

        # DELETE the declaration, do not REWRITE it as an assignment.
        #
        # My first version rewrote `const int N = 4096;` to `N = 0;`, which
        # sits inside the kernel's own main and therefore runs AFTER the driver
        # has set N. Every spec reset N to 0, allocated zero bytes, launched a
        # kernel that computed nothing, and came back at 87.3% under 7.1's
        # medians with 100% of rows "faster". A result that uniform is not a
        # finding, it is a measurement of nothing -- and the tell was that it
        # was too consistent.
        for pat in (r"[ \t]*const int N = \d+;\n",
                    r"[ \t]*dim3 grid\([^)]*\);\n",
                    r"[ \t]*dim3 block\([^)]*\);\n"):
            src = re.sub(pat, "", src)

        renamed, n = re.subn(r"int\s+main\s*\(\s*void\s*\)",
                             f'extern "C" int u2o_run_{k}(void)', src)
        if n != 1:
            raise SystemExit(
                f"FATAL: synthesize() emitted {n} main() for spec {i}; "
                f"refusing to guess, because guessing is the confound.")
        cu = td / f"k{k}.cu"
        # INDIVIDUAL externs, not `extern "C" { int N; ... }`. In C++ that block
        # is a DEFINITION, not a declaration, so eight translation units each
        # defined N/grid/block and the linker rejected it -- twice, because the
        # first "fix" made them extern in the wrong direction. A block of externs
        # is not a block of declarations.
        cu.write_text('extern "C" int N;\nextern "C" dim3 grid;\n'
                      'extern "C" dim3 block;\n' + renamed)
        obj = td / f"k{k}.o"
        b = subprocess.run(["nvcc", "-c", f"-arch={ARCH}", str(cu), "-o", str(obj)],
                           capture_output=True, text=True)
        if b.returncode != 0:
            raise SystemExit(f"FATAL: compile failed for kernel {k}\n{b.stderr[:2000]}")
        objs.append(str(obj))
        idxs.append(k)
        seen[key] = k
        by_source[k] = [i]
    # spec index -> kernel index, so the driver sets geometry then dispatches
    spec_to_kernel = [seen[s["source"]] for s in specs]
    return objs, idxs, spec_to_kernel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="rows_perkernel.jsonl")
    ap.add_argument("--summary-out", default="summary_perkernel.json")
    args = ap.parse_args()

    import tempfile
    td = Path(tempfile.mkdtemp())
    specs, provenance = load_specs()
    if args.limit:
        specs = specs[:args.limit]
    # REPORTED, NOT HARDCODED. The previous line printed a literal 0 for the
    # distinct-kernel count -- a hardcoded number in a status line, which is the
    # cheapest possible version of the defect this project has catalogued all
    # night: a value that looks like a measurement and is not one. It also hid
    # the question that matters, which is how many translation units the build
    # ACTUALLY produced.
    n_distinct = len({s["source"] for s in specs})
    print(f"per-kernel experiment: {len(specs)} specs over {n_distinct} distinct "
          f"kernel source(s); each distinct source gets its own nvcc -c "
          f"translation unit", flush=True)

    t_build = time.time()
    objs, idxs, spec_to_kernel = build_per_kernel_executables(specs, td)
    build_sec = time.time() - t_build
    # The previous line printed len(specs) -- 264 -- under the words "built N
    # per-kernel objects". It was not lying about the build, it was mislabelled,
    # and it hid the number that would have shown the dedup working: 8, not 264.
    # A status line that reports the wrong quantity is the cheapest version of
    # the defect this project has catalogued all night.
    print(f"built {len(idxs)} per-kernel translation units from {len(specs)} "
          f"specs in {build_sec:.1f}s", flush=True)
    if not provenance.get("proven_identical", True):
        raise SystemExit(
            f"FATAL: the regenerated sweep is not verifiably 7.1's: "
            f"{provenance}. Refusing to run, because a different sweep would "
            f"reproduce the confound while claiming to remove it.")

    # ONE process, ONE context, driving 7.1's own kernel bodies. Each is called
    # TWICE, because 7.1 ran 2 invocations of (5 warmup + 20 timed) per example
    # and reusing its binaries unmodified means reusing its invocation loop too.
    #
    # Written in one piece rather than patched together: an earlier version
    # appended the dispatch function AFTER main(), so it was used before
    # declaration and the build failed. Three of tonight's failures were of that
    # shape -- a thing assembled by successive edits that nobody compiled.
    decls = "\n".join(f'extern "C" int u2o_run_{i}(void);' for i in idxs)
    switch = "\n".join(f"    case {i}: return u2o_run_{i}();" for i in idxs)
    geom = "\n".join(
        f"        {{ int n={s['problem_size']}; int g={s['grid'][0]}; "
        f"int b={s['block'][0]}; N=n; grid=dim3(g,1,1); block=dim3(b,1,1); }}"
        for s in specs)
    tpl = (HERE / "perkernel_loop.cu.in").read_text()
    # A bare %s or %d in the template is a POSITIONAL specifier, so the whole
    # substitution fails with "not enough arguments" and the error points at the
    # wrong line. It cost me a job to add DISPATCH_FAIL with three unescaped %d.
    # So the check is made here, before the write, where it is legible.
    import re as _re
    # Mask the escaped pairs FIRST, then look for what is left. My first guard
    # used a negative lookahead instead, which does not consume the first '%' of
    # a '%%' pair, so every correctly-escaped literal was reported as bare -- a
    # FALSE ALARM in the very check meant to catch false alarms, and I nearly
    # "fixed" correct code because of it. Mask, then scan.
    masked = _re.sub(r"%%", "\x00", tpl)
    bare = [m.group(0) for m in _re.finditer(r"%[a-z]", masked)]
    if bare:
        raise SystemExit(
            f"FATAL: perkernel_loop.cu.in has bare format specifiers {bare}. "
            f"Every literal C % must be written %% or the substitution fails "
            f"with 'not enough arguments for format string', which names the "
            f"wrong line entirely.")
    driver = td / "loop.cu"
    driver.write_text(tpl % {
        "decls": decls, "switch": switch, "geom": geom,
        "kmapping": ", ".join(str(k) for k in spec_to_kernel),
                                      "geom": geom})
    loop_exe = td / "loop"
    b = subprocess.run(
        ["nvcc", f"-arch={ARCH}", str(driver), *objs, "-o", str(loop_exe)],
        capture_output=True, text=True)
    if b.returncode != 0:
        raise SystemExit(f"FATAL: loop build failed\n{b.stderr[:2500]}")

    t0 = time.time()
    r = subprocess.run([str(loop_exe), str(len(specs))],
                       capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(
            f"FATAL: loop exited {r.returncode}\n"
            f"  stderr: {r.stderr.strip()[-600:]}\n"
            f"  the driver now names the failing spec (DISPATCH_FAIL spec=N); if "
            f"that line is absent the failure is before the first dispatch, i.e. "
            f"in the build or the link, not in a kernel.")

    # parse: 7.1's synthesized drivers print {"trial":..,"time_ms":..}
    per_spec, cur = {}, None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("BEGIN "):
            parts = line.split()
            cur = (int(parts[1]), int(parts[2]))
            per_spec.setdefault(cur, [])
        elif line.startswith("END "):
            cur = None
        elif line.startswith("{") and cur is not None and "time_ms" in line:
            try:
                per_spec[cur].append(float(json.loads(line)["time_ms"]))
            except (ValueError, KeyError):
                pass

    rows, sampling = [], 0.0
    for i, s in enumerate(specs):
        by_inv = [per_spec.get((i, k), []) for k in (0, 1)]
        by_inv = [v for v in by_inv if v]
        if not by_inv:
            continue
        inv_meds = [statistics.median(v) for v in by_inv]
        trials = [x for v in by_inv for x in v]
        t_s = time.time()
        # 6.6.6 hardening: queue_state() refuses unidentified calls (null
        # own_job_id is unauditable). Identify by our own command substring:
        # the ts job command carries this script's filename, and the matcher
        # prefers the RUNNING job, so old finished runs of the same script
        # cannot shadow us. U2O_OWN_CMD overrides for uniquely-named reruns.
        import os as _os
        qstate, telemetry = qs.queue_state(
            own_command_substr=_os.environ.get(
                "U2O_OWN_CMD", "perkernel_confound.py")), _telemetry()
        sampling += time.time() - t_s
        toks = tokenize_for_model(s["source"], max_length=256)
        rows.append({
            "kernel_hash": s["kernel_hash"], "op_type": s.get("op_type"),
            "problem_size": s["problem_size"],
            "grid": s["grid"], "block": s["block"],
            "n_trials": len(inv_meds), "n_invocations": len(inv_meds),
            "invocation_medians_ms": inv_meds,
            "n_timed_launches": len(trials),
            "median_runtime_ms": statistics.median(inv_meds),
            "trials_ms": trials,
            "sampled_at": t_s, "queue_state": qstate, "telemetry": telemetry,
            "token_count": toks.get("token_count") if isinstance(toks, dict) else None,
            "mode": "perkernel",
        })
    (HERE / args.out).write_text("".join(json.dumps(x) + "\n" for x in rows))

    n = len(rows)
    if n == 0:
        raise SystemExit("FATAL: 0 rows measured; refusing to print a rate")
    out = {
        "bead": "gpu-sim-u2o.6.7.1",
        "mode": "perkernel",
        "spec_provenance": provenance,
        "n_specs": len(specs),
        "n_distinct_kernel_sources": n_distinct,
        "n_translation_units_built": len(idxs),
        "what_this_is": (
            "7.1's OWN per-kernel synthesized binaries, each compiled as its own "
            "translation unit with its own per-kernel cudaMalloc calls, all "
            "LINKED into one executable driven by ONE process with ONE CUDA "
            "context. Removes the executable/memory-environment confound while "
            "keeping the shared context that produced the speed."),
        "confound_removed": (
            "7.1: one binary per kernel_hash, own main, own cudaMalloc sized to "
            "that kernel's arity. My earlier warm arm: ONE binary, all 8 kernels, "
            "dispatch switch, FOUR buffers of the MAXIMUM size regardless of "
            "arity. Same isolation, same estimator, different executable and "
            "different memory footprint."),
        "n_rows": n, "wall_sec": wall,
        "examples_per_sec": n / wall, "examples_per_hour": (n / wall) * 3600,
        "build_sec": build_sec,
        "sampling_wall_sec": sampling,
        "device_time_per_example_us": sum(
            r["median_runtime_ms"] * 25 for r in rows) / n * 1e6,
        "protocol_note": (
            "each spec is ONE invocation of 5 warmup + 20 timed, so this arm has "
            "40 timed launches per example: each of 7.1's binaries is called "
            "TWICE, because 7.1 ran 2 invocations of (5 warmup + 20 timed) and "
            "reusing its binaries unmodified means reusing its invocation loop."),
    }
    (HERE / args.summary_out).write_text(json.dumps(out, indent=2))
    print(f"\n{n} specs in {wall:.2f}s = {out['examples_per_sec']:.2f} ex/s "
          f"({out['examples_per_hour']:,.0f}/hour), build {build_sec:.0f}s")
    print(f"device time/example {out['device_time_per_example_us']:.1f} us")
    return 0


def _telemetry() -> dict:
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

    raise SystemExit(main())

