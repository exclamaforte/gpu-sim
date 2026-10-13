#!/usr/bin/env python3
"""Per-op-class latency lookup for CI/behavioral simulation (13.4).

Env/config-driven table (never hardcoded): GPUSIM_OP_LATENCY_JSON points
at op_latency_table.json; unknown ops fall back to the legacy default.
D_GPUSIM: simulation timing only, never training ground truth.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = REPO / "experiments/gpu-sim-u2o.13.4/op_latency_table.json"
LEGACY_DEFAULT_MS = 0.5

_cache = {}


def load_table(path=None):
    p = Path(path or os.environ.get("GPUSIM_OP_LATENCY_JSON", DEFAULT_TABLE))
    key = str(p)
    if key not in _cache:
        _cache[key] = json.loads(p.read_text())
    return _cache[key]


def latency_ms_for(op, table=None, default=LEGACY_DEFAULT_MS) -> float:
    """Calibrated median for a known op class, else the legacy default."""
    if table is None:
        try:
            table = load_table()
        except (FileNotFoundError, json.JSONDecodeError):
            return default
    ops = table.get("ops", {}) if isinstance(table, dict) else {}
    return float(ops[op]["median_ms"]) if op in ops else default


def latency_ms_for_spec(spec, table=None) -> float:
    """Harness wiring: spec dict -> latency (spec.get('op_type'))."""
    return latency_ms_for((spec or {}).get("op_type"), table)


if __name__ == "__main__":
    print(f"op_latency OK: relu={latency_ms_for('relu'):.4f} ms, "
          f"unknown={latency_ms_for('nope')}")
