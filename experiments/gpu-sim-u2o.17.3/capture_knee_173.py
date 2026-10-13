#!/usr/bin/env python3
"""gpu-sim-u2o.17.3 knee dose (ONE ts job): elementwise cases at 2^21 /
2^22 / 2^23, bracketing the 1M->16M launch-floor knee. Protocol IDENTICAL
to 17.5 (imports capture_fixture.capture_case/telemetry/LEAF/WARMUP/REPS
-- zero drift by construction). Supplement fixture, not a rewrite.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/gpu-sim-u2o.17.5"))

import torch  # noqa: E402

import capture_fixture as CF  # noqa: E402

SIZES = (1 << 21, 1 << 22, 1 << 23)
OPS = (("relu", torch.relu, "aten::relu"),
       ("sigmoid", torch.sigmoid, "aten::sigmoid"),
       ("tanh", torch.tanh, "aten::tanh"))


def build_knee_cases():
    cases = []
    for n in SIZES:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        for op, fn, want in OPS:
            cases.append(({
                "op_class": "elementwise", "op": op, "shapes": [n],
                "dtype": "torch.float32", "want": want,
                "note": "17.3 knee dose: brackets the 1M->16M floor knee",
            }, (lambda x=x, fn=fn: fn(x))))
    return cases


def main() -> int:
    if "--list" in sys.argv:
        print(f"{len(SIZES) * len(OPS)} knee cases (dry list, no CUDA):")
        for n in SIZES:
            for op, _, _ in OPS:
                print(f"  elementwise {op} [{n}]")
        return 0
    if "--self-test" in sys.argv:
        torch.cuda.init()
        print("UNEXPECTED: knee capture succeeded with no CUDA device")
        return 0
    if not torch.cuda.is_available():
        print("FATAL: no CUDA device; refusing to write knee cases")
        return 1
    torch.manual_seed(0)
    trace_dir = HERE / "traces_knee"
    trace_dir.mkdir(exist_ok=True)
    tel_before = CF.telemetry("before")
    props = torch.cuda.get_device_properties(0)
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        drv = f"error: {e}"
    cases = build_knee_cases()
    total = len(cases)
    records = []
    for i, (desc, thunk) in enumerate(cases):
        rec = CF.capture_case(i, desc, thunk, trace_dir)
        if "note" in desc:
            rec["note"] = desc["note"]
        case_tel = CF.telemetry(f"case{i}")
        rec["telemetry"] = case_tel.get("fields", case_tel)
        rec["throttle_active"] = (rec["telemetry"].get(
            "clocks_throttle_reasons.active", "0x0")
            not in ("0x0000000000000000", "", "Not Supported"))
        records.append(rec)
        print(f"GPQUEUE_PROGRESS done={i + 1} total={total} unit=cases",
              flush=True)
        if "error" in rec:
            print(f"  !! {rec['op']} {rec['shapes']} {rec['error']}",
                  flush=True)
        else:
            print(f"  {rec['op']:8s} {str(rec['shapes']):12s} "
                  f"median={rec['device_ms_median'] * 1000:.3f}us",
                  flush=True)
    tel_after = CF.telemetry("after")
    fix = {
        "bead": "gpu-sim-u2o.17.3",
        "what": "knee-dose supplement: elementwise 2M/4M/8M (17.5 protocol)",
        "device": {"name": props.name,
                   "driver_version": drv,
                   "torch_version": torch.__version__},
        "telemetry_before": tel_before,
        "telemetry_after": tel_after,
        "n_cases": len(records),
        "cases": records,
    }
    (HERE / "knee_cases.json").write_text(json.dumps(fix, indent=1))
    print(f"knee dose complete: {len(records)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
