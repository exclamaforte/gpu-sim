#!/usr/bin/env python3
"""GPU-spec conditioning vectors for cross-GPU generalization (13.2).

Maps the sourced hardware constants in hw_5090.py to a fixed-order,
unit-documented conditioning vector. Missing fields raise (no silent
zeros); each entry carries its provenance tag. A second GPU = a second
entry + its measured rows (gate test asserts that data is absent today).
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from epic_u2o import hw_5090 as hw

FIELDS = (
    # (key, unit, source-note)
    ("sm_count", "count", "hw_5090.SM_COUNT (XLAB+OWN)"),
    ("peak_fp32_tflops", "TFLOP/s", "hw_5090.PEAK_FP32_TFLOPS "
     "(VENDOR, UNSOURCED IN-TREE -- carried, not laundered)"),
    ("mem_bw_gbs", "GB/s", "hw_5090.THEORETICAL_READ_BW_GBS (VENDOR)"),
    ("l2_bytes", "bytes", "hw_5090.L2_CACHE_BYTES (OWN+XLAB)"),
    ("shared_mem_per_sm_bytes", "bytes",
     "hw_5090.SHARED_MEM_PER_SM_BYTES (OWN+XLAB)"),
    ("max_sm_clock_mhz", "MHz", "hw_5090.MAX_SM_CLOCK_MHZ (XLAB)"),
)

SPECS = {
    "rtx5090": {
        "sm_count": hw.SM_COUNT,
        "peak_fp32_tflops": hw.PEAK_FP32_TFLOPS,
        "mem_bw_gbs": hw.THEORETICAL_READ_BW_GBS,
        "l2_bytes": hw.L2_CACHE_BYTES,
        "shared_mem_per_sm_bytes": hw.SHARED_MEM_PER_SM_BYTES,
        "max_sm_clock_mhz": hw.MAX_SM_CLOCK_MHZ,
    },
}


def vectorize(name: str) -> list:
    """Fixed-order conditioning vector; unknown GPU or missing field raises."""
    if name not in SPECS:
        raise KeyError(f"no spec for {name!r}; add measured constants first")
    spec = SPECS[name]
    vec = []
    for key, unit, _src in FIELDS:
        if key not in spec or spec[key] is None:
            raise ValueError(f"spec {name!r} missing field {key} ({unit})")
        vec.append(float(spec[key]))
    return vec


def field_docs() -> list:
    return [{"field": k, "unit": u, "source": s} for k, u, s in FIELDS]


if __name__ == "__main__":
    v = vectorize("rtx5090")
    assert len(v) == len(FIELDS) and all(x > 0 for x in v)
    print(f"gpu_spec OK: rtx5090 dim={len(v)} sm={v[0]:.0f}")
