#!/usr/bin/env python3
"""Per-op-class (and size-aware) latency lookup for CI/behavioral simulation (13.4, 9ei).

Validity envelope (gpu-sim-v4e, experiments/gpu-sim-v4e/REPORT.md):
measured on RTX 5090 against the m93 calibration table, no refit.
  - Below the grid: elementwise holds 3 octaves, reduce ~3 (marginal, max 0.236),
    matmul only 1 octave (fails at 2).
  - Above the grid: 0 octaves supported for every class; error 46-95% because the
    clamp holds the prediction flat while measured cost keeps growing.
Out-of-range requests are clamped to the nearest calibrated size and a warning
is emitted once per (op, direction). Callers can inspect prediction_provenance()
to distinguish a measured prediction from a clamped one.


Env/config-driven table (never hardcoded): GPUSIM_OP_LATENCY_JSON points
at a table (13.4 schema or mdz calibration schema); unknown ops fall back to
the legacy default. D_GPUSIM: simulation timing only, never training ground truth.
"""
import json
import warnings
import math
import os
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = REPO / "experiments/gpu-sim-u2o.13.4/op_latency_table.json"
LEGACY_DEFAULT_MS = 0.5

_cache = {}
_warned = set()

def _warn_once(op, direction):
    key = (op, direction)
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(f"op_latency: {op} size out of calibrated range ({direction}); "
                  f"prediction is clamped to the table edge and may be inaccurate.",
                  stacklevel=3)


def _normalize_table(raw):
    """Normalize 13.4 or mdz calibration schema into 13.4 schema with optional sizes."""
    if isinstance(raw, dict) and "cases" in raw:
        ops = {}
        for c in raw["cases"]:
            if "error" in c:
                continue
            op = c.get("op", c.get("op_class"))
            sh = c.get("shapes", [c.get("shape")])
            if isinstance(sh, list) and sh and isinstance(sh[0], list):
                sh = sh[0]
            size = sh[0] if isinstance(sh, list) else sh
            if "device_ms_median" in c:
                ms = c["device_ms_median"]
            elif "median_us" in c:
                ms = c["median_us"] / 1000.0
            else:
                continue
            entry = ops.setdefault(op, {"median_ms": ms, "n": 0, "sizes": {}, "bytes_per_element": 4})
            entry["n"] = entry.get("n", 0) + 1
            entry["sizes"][str(size)] = ms
            # keep overall median as median of size medians
        # fix median import
        import statistics
        for op, e in ops.items():
            e["median_ms"] = statistics.median(list(e["sizes"].values()))
        return {"ops": ops}
    return raw


def load_table(path=None):
    p = Path(path or os.environ.get("GPUSIM_OP_LATENCY_JSON", DEFAULT_TABLE))
    key = str(p)
    if key not in _cache:
        raw = json.loads(p.read_text())
        _cache[key] = _normalize_table(raw)
    return _cache[key]


def _size_from_spec(spec):
    if not isinstance(spec, dict):
        return None
    for k in ("size", "n", "num_elements"):
        if k in spec:
            return spec[k]
    sh = spec.get("shape") or spec.get("shapes")
    if isinstance(sh, (list, tuple)) and sh:
        first = sh[0]
        if isinstance(first, (list, tuple)):
            first = first[0]
        try:
            return int(first)
        except (TypeError, ValueError):
            return None
    return None



def prediction_provenance(op, size=None, table=None, dtype_bytes=4):
    """Return provenance for a (op, size) prediction.

    Keys: clamped (bool), clamp_target (int or None), cal_min (int or None),
    cal_max (int or None). clamped is True when size falls outside the
    calibrated range and the prediction is the edge value.
    """
    if isinstance(size, dict) and table is None:
        table = size
        size = None
    if table is None:
        try:
            table = load_table()
        except (FileNotFoundError, json.JSONDecodeError):
            return {"clamped": False, "clamp_target": None,
                    "cal_min": None, "cal_max": None,
                "no_size_axis": False}
    ops = table.get("ops", {}) if isinstance(table, dict) else {}
    if op not in ops:
        return {"clamped": False, "clamp_target": None,
                "cal_min": None, "cal_max": None,
                "no_size_axis": False}
    entry = ops[op]
    bpe = entry.get("bytes_per_element", 4) if isinstance(entry, dict) else 4
    if size is not None:
        size = size * dtype_bytes / bpe
        if isinstance(size, float) and size.is_integer():
            size = int(size)
    if not (isinstance(entry, dict) and "sizes" in entry):
        return {"clamped": None, "clamp_target": None,
                "cal_min": None, "cal_max": None,
                "no_size_axis": True}
    try:
        sizes = sorted(entry["sizes"], key=lambda x: int(x))
        xs = [int(x) for x in sizes]
    except (ValueError, TypeError):
        return {"clamped": False, "clamp_target": None,
                "cal_min": None, "cal_max": None,
                "no_size_axis": False}
    if not xs:
        return {"clamped": False, "clamp_target": None,
                "cal_min": None, "cal_max": None,
                "no_size_axis": False}
    cal_min, cal_max = xs[0], xs[-1]
    if size is None:
        return {"clamped": False, "clamp_target": None,
                "cal_min": cal_min, "cal_max": cal_max,
                "no_size_axis": False}
    try:
        si = int(size)
    except (TypeError, ValueError):
        return {"clamped": False, "clamp_target": None,
                "cal_min": cal_min, "cal_max": cal_max,
                "no_size_axis": False}
    if si < cal_min:
        return {"clamped": True, "clamp_target": cal_min,
                "cal_min": cal_min, "cal_max": cal_max,
                "no_size_axis": False}
    if si > cal_max:
        return {"clamped": True, "clamp_target": cal_max,
                "cal_min": cal_min, "cal_max": cal_max,
                "no_size_axis": False}
    return {"clamped": False, "clamp_target": None,
            "cal_min": cal_min, "cal_max": cal_max,
            "no_size_axis": False}

def latency_ms_for(op, size=None, table=None, dtype_bytes=4, default=LEGACY_DEFAULT_MS) -> float:
    """Calibrated median for a known op class, size-aware when possible."""
    if isinstance(size, dict) and table is None:
        table = size
        size = None
    if table is None:
        try:
            table = load_table()
        except (FileNotFoundError, json.JSONDecodeError):
            return default
    ops = table.get("ops", {}) if isinstance(table, dict) else {}
    if op not in ops:
        return default
    entry = ops[op]
    bpe = entry.get("bytes_per_element", 4) if isinstance(entry, dict) else 4
    if size is not None:
        size = size * dtype_bytes / bpe
        if isinstance(size, float) and size.is_integer():
            size = int(size)
    if size is not None and isinstance(entry, dict) and "sizes" in entry:
        key = str(size)
        if key in entry["sizes"]:
            return float(entry["sizes"][key])
        # log-log interpolation between bracketing sizes (matches m93 --calibration);
        # clamp at the ends if out of range (stated rule).
        try:
            sizes = sorted(entry["sizes"], key=lambda x: int(x))
            xs = [int(x) for x in sizes]
            ys = [float(entry["sizes"][x]) for x in sizes]
            if not sizes:
                return float(entry.get("median_ms", default))
            if int(size) <= xs[0]:
                _warn_once(op, "below")
                return ys[0]
            if int(size) >= xs[-1]:
                _warn_once(op, "above")
                return ys[-1]
            for i in range(len(xs)-1):
                if xs[i] <= int(size) <= xs[i+1]:
                    x0,y0=xs[i],ys[i]; x1,y1=xs[i+1],ys[i+1]
                    t=(math.log(int(size))-math.log(x0))/(math.log(x1)-math.log(x0))
                    return math.exp(math.log(y0)+t*(math.log(y1)-math.log(y0)))
        except (ValueError, ZeroDivisionError):
            pass
    return float(entry.get("median_ms", default)) if isinstance(entry, dict) else float(entry)


def latency_ms_for_spec(spec, table=None) -> float:
    """Harness wiring: spec dict -> latency (spec.get('op_type'), size-aware, dtype-aware)."""
    spec = spec or {}
    dtype_bytes = spec.get("dtype_bytes", 4)
    return latency_ms_for(spec.get("op_type"), size=_size_from_spec(spec), dtype_bytes=dtype_bytes, table=table)


if __name__ == "__main__":
    print(f"op_latency OK: relu={latency_ms_for('relu'):.4f} ms, "
          f"unknown={latency_ms_for('nope')}")
