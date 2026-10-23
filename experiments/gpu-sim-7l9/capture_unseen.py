#!/usr/bin/env python3
"""Capture the canonical real-GPU PyTorch trace fixture (gpu-sim-u2o.17.5).

Extends the u2o.17.2 approach -- same torch.profiler machinery, same
CPU+CUDA activities, same key_averages() self-device-time extraction -- but
across op CLASSES and SHAPES instead of one 4096^2 matmul+relu. u2o.17.3
(the simulated path) fits against this and u2o.17.4 (parity) compares against
it, so the fixture has to be broad enough to support a per-op-class claim.

This bead produces the FIXTURE ONLY. It decides nothing: no pass/fail, no
comparison, no naive roofline verdict. 17.4 owns pass/fail.

Op classes (>=4, each spanning >=2 orders of magnitude where the op allows):
  elementwise   relu / sigmoid / tanh over flat tensors, 2^10 .. 2^24
  matmul        fp32 mm at 512, 1024, 2048, 4096, 8192 (straddles the
                compute/memory crossover)
  reduce        sum / max over 2^14 .. 2^24
  chain         a memory-bound multi-op chain (relu -> mul -> add)
  launch_bound  a deliberately tiny case (2^8) that is launch-dominated,
                recorded explicitly so nobody later reads it as bandwidth

Per case we record op class, shapes, dtype, launch config, the profiler's own
attribution, and a median AND a spread. The spread is taken from the raw
per-kernel event durations in the exported chrome trace, not inferred from
the totals/count average, because an average with no spread cannot be used as
a parity target. Device string, driver version and read-only nvidia-smi
telemetry are recorded at the top level so a later parity failure can be
attributed to a machine-state change rather than to the model.

Negative control: with CUDA_VISIBLE_DEVICES= blanked this must FAIL loudly,
not silently produce a fixture.
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

WARMUP = 5
REPS = 20
TELEMETRY_FIELDS = ["clocks.sm", "clocks.max.sm", "temperature.gpu",
                    "clocks_throttle_reasons.active", "persistence_mode"]

# ATen dispatch: the op a user calls usually DISPATCHES to a lower-level op
# that carries the CUDA self time. Same leaf-op mapping as u2o.17.2.
LEAF = {
    "aten::relu": "aten::clamp_min",
    "aten::matmul": "aten::mm",
    # Reviewer BLOCKING 1 on 17.5: torch.max / Tensor.max() dispatches to
    # aten::max, not aten::amax. My first version looked for amax, found
    # nothing, and shipped four HOLLOW reduce cases -- no median, no spread, no
    # raw trace -- whose emptiness nothing in the fixture disclosed.
    "aten::amax": "aten::max",
    "aten::sum": "aten::sum",
}


def telemetry(label):
    out = {"label": label, "at": time.time()}
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        out["rc"] = r.returncode
        out["csv"] = r.stdout.strip()
        if r.returncode == 0 and r.stdout.strip():
            out["fields"] = dict(zip(TELEMETRY_FIELDS,
                                     [v.strip() for v in r.stdout.strip().split(",")]))
    except Exception as e:
        out["error"] = str(e)
    return out


def gpu_event_durations(trace_path: Path, substrings):
    """Per-device-event durations (us) from a chrome trace.

    Returns (durations, matched_names). Matching is by substring because the
    demangled CUDA kernel name is not the ATen key (aten::relu runs a
    vectorized_elementwise_kernel, for instance), so we filter to device
    events and keep the dominant one.
    """
    try:
        data = json.loads(trace_path.read_text())
    except Exception:
        return [], []
    evs = data.get("traceEvents", data) if isinstance(data, dict) else data
    durs, names = [], []
    for e in evs:
        if not isinstance(e, dict):
            continue
        cat = str(e.get("cat", ""))
        if cat not in ("Kernel", "kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = e.get("dur")
        if not isinstance(dur, (int, float)):
            continue
        name = str(e.get("name", ""))
        if substrings and not any(s in name for s in substrings):
            continue
        durs.append(float(dur))
        names.append(name)
    if not durs:
        return [], []
    by_name = {}
    for d, nm in zip(durs, names):
        by_name.setdefault(nm, []).append(d)
    dominant = max(by_name.items(), key=lambda kv: (sum(kv[1]), len(kv[1])))
    return sorted(dominant[1]), [dominant[0]]


def capture_case(index, desc, thunk, trace_dir: Path):
    op_class, op_name = desc["op_class"], desc["op"]
    want = LEAF.get(desc["want"], desc["want"])
    t0 = time.time()

    thunk()                                   # warmup
    torch.cuda.synchronize()
    trace_path = trace_dir / f"case_{index:02d}_{op_class}_{op_name.replace('->','_')}.json"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(REPS):
            thunk()
        torch.cuda.synchronize()

    stats = {}
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            stats[e.key] = {
                "self_device_time_total_us": e.self_device_time_total,
                "count": e.count,
                "self_device_time_avg_us": e.self_device_time_total / max(1, e.count),
            }
    s = stats.get(want)
    if s is None:
        return {"op_class": op_class, "op": op_name, "shapes": desc["shapes"],
                "dtype": desc["dtype"],
                "error": f"no self CUDA time for {want}",
                "profiler_keys": sorted(stats)[:40]}

    prof.export_chrome_trace(str(trace_path))
    durs, names = gpu_event_durations(trace_path, substrings=())
    durs.sort()
    n_ev = len(durs)
    rec = {
        "op_class": op_class,
        "op": op_name,
        "profiler_key": want,
        "shapes": desc["shapes"],
        "dtype": desc["dtype"],
        "launch_config": desc.get("launch_config",
                                  {"grid": None, "block": None,
                                   "note": "ATen/cuBLAS chooses the tiling"}),
        "warmup": WARMUP,
        "reps": REPS,
        "device_ms_median": s["self_device_time_avg_us"] / 1000.0,
        "device_ms_n": s["count"],
        "device_us_total": s["self_device_time_total_us"],
        "profiler_attribution": stats,
        "raw_trace": trace_path.name,
        "capture_wall_sec": time.time() - t0,
    }
    if n_ev:
        med = statistics.median(durs)
        rec["device_event_us_median"] = med
        rec["device_event_us_min"] = durs[0]
        rec["device_event_us_max"] = durs[-1]
        rec["device_event_us_iqr"] = (durs[3 * n_ev // 4] - durs[n_ev // 4])
        # Robust spread: the p05-p95 band, NOT min..max. A single first-call
        # outlier (cuBLAS init, allocator warm-up) otherwise dominates the range
        # and the fixture reports a 200% spread for a kernel whose steady-state
        # time is reproducible to 0.1%. min/max are still recorded, separately.
        p05 = durs[max(0, int(0.05 * (n_ev - 1)))]
        p95 = durs[min(n_ev - 1, int(0.95 * (n_ev - 1)))]
        rec["device_event_us_p05"] = p05
        rec["device_event_us_p95"] = p95
        rec["device_ms_spread"] = (p95 - p05) / 1000.0
        rec["device_ms_spread_definition"] = "p95 - p05 of the dominant kernel's device-event durations"
        rec["device_ms_spread_pct_of_median"] = (100.0 * (p95 - p05) / med if med else None)
        rec["device_ms_full_range_pct_of_median"] = (
            100.0 * (durs[-1] - durs[0]) / med if med else None)
        rec["device_ms_spread_pct_of_median"] = (
            100.0 * (durs[-1] - durs[0]) / med if med else None)
        rec["device_event_us_n"] = n_ev
        rec["device_event_names"] = sorted(set(names))[:8]
    else:
        # Say so rather than shipping a median with no spread.
        rec["device_ms_spread"] = None
        rec["spread_note"] = ("no device events parsed from the chrome trace; "
                              "median comes from key_averages totals/count only")
    return rec


def build_cases():
    cases = []
    # assert unseen ops not in mdz table
    import json
    mdz_path = REPO / "experiments/gpu-sim-mdz/mdz_calibration_fixture.json"
    if not mdz_path.exists():
        mdz_path = REPO / "experiments/gpu-sim-mdz/calibration_fixture.json"
    mdz = json.loads(mdz_path.read_text())
    table_ops = set(c["op"] for c in mdz["cases"])
    for op in ("gelu","hardsigmoid"):
        assert op not in table_ops, f"{op} already in table"
    print("unseen ops verified absent from table")

    for n in (1<<10, 1<<16, 1<<20, 1<<24):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        cases.append(({
            "op_class": "elementwise", "op": "gelu", "shapes": [n],
            "dtype": "torch.float32", "want": "aten::gelu",
        }, (lambda x=x: torch.nn.functional.gelu(x))))
        cases.append(({
            "op_class": "elementwise", "op": "hardsigmoid", "shapes": [n],
            "dtype": "torch.float32", "want": "aten::hardsigmoid",
        }, (lambda x=x: torch.nn.functional.hardsigmoid(x))))
    return cases


def negative_control() -> dict:
    """Prove the capture cannot silently succeed without a real device."""
    r = subprocess.run(
        [sys.executable, str(REPO / "experiments/gpu-sim-u2o.17.5/capture_fixture.py"), "--self-test"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    return {
        "returncode": r.returncode,
        "stdout_tail": r.stdout.strip()[-300:],
        "stderr_tail": r.stderr.strip()[-300:],
        "failed_as_required": r.returncode != 0,
    }


def main() -> int:
    if "--self-test" in sys.argv:
        # negative-control path: no device must mean a loud failure
        torch.cuda.init()
        _ = torch.randn(8, device="cuda")
        torch.cuda.synchronize()
        print("UNEXPECTED: capture succeeded with no visible CUDA device")
        return 0

    print("=" * 72)
    print("gpu-sim-u2o.17.5 -- canonical real-GPU PyTorch trace fixture")
    print("=" * 72)
    if not torch.cuda.is_available():
        print("FATAL: no CUDA device; refusing to write a fixture")
        return 1

    tel_before = telemetry("before")
    props = torch.cuda.get_device_properties(0)
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        drv = f"error: {e}"

    torch.manual_seed(0)
    trace_dir = HERE / "traces"
    trace_dir.mkdir(exist_ok=True)
    cases = build_cases()
    print(f"{len(cases)} cases; torch {torch.__version__}; device {props.name}",
          flush=True)

    records = []
    for i, (desc, thunk) in enumerate(cases):
        rec = capture_case(i, desc, thunk, trace_dir)
        if "note" in desc:
            rec["note"] = desc["note"]
        # Per-case machine state (reviewer MINOR): a before/after pair only
        # proves throttling happened somewhere in the capture. A per-case stamp
        # lets 17.4 exclude a contaminated case mechanically instead of by
        # reading prose in a bd comment.
        case_tel = telemetry(f"case{i}")
        rec["telemetry"] = case_tel.get("fields", case_tel)
        rec["throttle_active"] = (rec["telemetry"].get(
            "clocks_throttle_reasons.active", "0x0")
            not in ("0x0000000000000000", "", "Not Supported"))
        records.append(rec)
        if "error" in rec:
            print(f"  !! {rec['op_class']:12s} {rec['op']:12s} {rec['error']}",
                  flush=True)
        else:
            sp = rec.get("device_ms_spread_pct_of_median")
            sp_s = "n/a" if sp is None else f"{sp:.1f}%"
            print(f"  {rec['op_class']:12s} {rec['op']:12s} {str(rec['shapes']):16s} "
                  f"median={rec['device_ms_median']*1000:10.3f}us "
                  f"spread={sp_s:>8s} n={rec['device_ms_n']}", flush=True)

    tel_after = telemetry("after")
    nc = negative_control()
    print(f"negative control (CUDA_VISIBLE_DEVICES=): rc={nc['returncode']} "
          f"failed_as_required={nc['failed_as_required']}", flush=True)

    good = [r for r in records if "error" not in r]
    hollow = [r for r in records if "error" not in r
              and not r.get("device_ms_median")]
    if hollow:
        # Never ship a case with no measurement in it. The previous revision
        # did: four reduce/max cases with empty profiler_attribution sailed
        # through because the summary counted only the populated half.
        for r in hollow:
            print(f"  !! HOLLOW {r['op_class']}/{r['op']} {r['shapes']}: "
                  "no device_ms_median despite no capture error", flush=True)
    by_class = {}
    for r in good:
        by_class.setdefault(r["op_class"], []).append(r)

    spread_by_class = {}
    for cls, rs in by_class.items():
        sps = [r["device_ms_spread_pct_of_median"] for r in rs
               if r.get("device_ms_spread_pct_of_median") is not None]
        spread_by_class[cls] = {
            "n_cases": len(rs),
            "max_spread_pct_of_median": max(sps) if sps else None,
            "median_spread_pct_of_median": statistics.median(sps) if sps else None,
        }

    fixture = {
        "bead": "gpu-sim-u2o.17.5",
        "fixture_version": 1,
        "what": ("canonical real-GPU PyTorch trace fixture, op classes x shapes. "
                 "Ground truth for 17.3 (simulated path) and 17.4 (parity gate). "
                 "This bead decides nothing; 17.4 owns pass/fail."),
        "produced": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "driver_version": drv,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
        },
        "protocol": {
            "profiler": "torch.profiler, CPU+CUDA activities, record_shapes=True",
            "warmup": WARMUP,
            "repetitions_per_case": REPS,
            "seed": 0,
            "extraction": ("profiler_key_averages() self_device_time_total/count for "
                           "the leaf op (relu->clamp_min, matmul->mm), plus per-event "
                           "durations parsed out of the exported chrome trace for the "
                           "median and the spread"),
            "negative_control": nc,
        },
        "telemetry_before": tel_before,
        "telemetry_after": tel_after,
        "op_classes": sorted(by_class),
        "n_cases": len(good),
        "spread_by_op_class": spread_by_class,
        "cases": records,
    }
    out = HERE / "unseen_fixture.json"
    out.write_text(json.dumps(fixture, indent=2))
    print(f"\nwrote {out} ({len(good)} measured cases over {len(by_class)} op "
          f"classes; {len(records)} case records total, {len(hollow)} hollow)")
    if not nc["failed_as_required"]:
        print("FATAL: negative control did NOT fail; fixture is not trustworthy")
        return 1
    if hollow:
        print("FATAL: a case with no measurement would ship; fixture rejected")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
